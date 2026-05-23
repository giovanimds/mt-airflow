from __future__ import annotations

import datetime
import json
import os
import re
import uuid
import pandas as pd
import scrapy
from itemadapter import ItemAdapter
from scrapy.crawler import CrawlerProcess
import fitz  # PyMuPDF


class ParquetChunkPipeline:
    def __init__(self, bucket_name: str | None, chunk_size: int = 50, local_fallback_dir: str = "./output"):
        self.bucket_name = bucket_name
        self.chunk_size = chunk_size
        self.local_fallback_dir = local_fallback_dir
        self.items: list[dict] = []
        self.chunk_count = 0
        self.spider_name = "unknown"
        
        if not self.bucket_name:
            os.makedirs(self.local_fallback_dir, exist_ok=True)
            print(f"Nenhum bucket especificado. Salvando localmente em: {os.path.abspath(self.local_fallback_dir)}")

    @classmethod
    def from_crawler(cls, crawler):
        bucket_name = os.environ.get("OUTPUT_BUCKET") or crawler.settings.get("OUTPUT_BUCKET")
        chunk_size = int(os.environ.get("CHUNK_SIZE", 50))
        local_fallback_dir = os.environ.get("LOCAL_OUTPUT_DIR", "./output")
        return cls(bucket_name, chunk_size, local_fallback_dir)

    def open_spider(self, spider):
        self.spider_name = spider.name

    def process_item(self, item, spider):
        self.spider_name = spider.name
        self.items.append(ItemAdapter(item).asdict())
        if len(self.items) >= self.chunk_size:
            self.write_chunk()
        return item

    def close_spider(self, spider):
        if self.items:
            self.write_chunk()

    def write_chunk(self):
        self.chunk_count += 1
        df = pd.DataFrame(self.items)
        self.items = []  # Clear accumulated items to prevent duplicates/leaks
        
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{self.spider_name}_{timestamp}_{unique_id}_chunk_{self.chunk_count}.parquet"
        
        if self.bucket_name:
            # GCS Path (gcsfs automatically uses GOOGLE_APPLICATION_CREDENTIALS)
            bucket_clean = self.bucket_name.replace("gs://", "").strip("/")
            gcs_path = f"gs://{bucket_clean}/raw_corpus/{filename}"
            print(f"Escrevendo chunk {self.chunk_count} com {len(df)} itens para o GCS: {gcs_path}")
            df.to_parquet(gcs_path, index=False)
        else:
            local_path = os.path.join(self.local_fallback_dir, filename)
            print(f"Escrevendo chunk {self.chunk_count} com {len(df)} itens localmente: {local_path}")
            df.to_parquet(local_path, index=False)


class WikipediaPTSpider(scrapy.Spider):
    name = "wikipedia_pt"
    allowed_domains = ["pt.wikipedia.org"]
    start_urls = [
        "https://pt.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0&prop=extracts&explaintext=1&format=json&grnlimit=1"
    ]

    def parse(self, response):
        try:
            data = json.loads(response.text)
        except Exception as e:
            self.logger.error(f"Erro ao ler JSON: {e}")
            return

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            title = page_info.get("title", "")
            text = page_info.get("extract", "").strip()

            if title and len(text) > 200:
                yield {
                    "title": title,
                    "text": text,
                    "url": f"https://pt.wikipedia.org/wiki/?curid={page_id}",
                    "language": "pt-br",
                    "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }

        # Continuar solicitando páginas aleatórias infinitamente (limitado pelo CLOSESPIDER_ITEMCOUNT global)
        yield scrapy.Request(
            self.start_urls[0],
            callback=self.parse,
            dont_filter=True
        )


class ArxivSpider(scrapy.Spider):
    name = "arxiv_pt"
    allowed_domains = ["export.arxiv.org", "arxiv.org"]
    
    # Query de exemplo, buscando artigos de CS
    start_urls = ["http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=50"]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 3.0,  # ArXiv requer delay
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'RETRY_HTTP_CODES': [429, 500, 502, 503, 504, 522, 524, 408],
        'RETRY_TIMES': 5,
    }

    def parse(self, response):
        response.meta["start"] = 0
        return self.parse_api(response)

    def parse_api(self, response):
        # ArXiv API returns XML (Atom format)
        entries = response.css("entry")
        
        if not entries:
            self.logger.warning("No more entries found or rate limit hit.")
            return

        for entry in entries:
            title = entry.css("title::text").get(default="").strip()
            pdf_url = entry.css("link[type='application/pdf']::attr(href)").get()
            
            if pdf_url:
                # O PDF url geralmente termina com 'v1', 'v2', etc. Mas podemos puxar direto.
                yield scrapy.Request(
                    pdf_url,
                    callback=self.parse_pdf,
                    meta={"title": title, "url": pdf_url}
                )
                
        # Next page
        start = response.meta["start"] + 50
        next_url = f"http://export.arxiv.org/api/query?search_query=cat:cs.AI&start={start}&max_results=50"
        yield scrapy.Request(next_url, callback=self.parse_api, meta={"start": start})

    def parse_pdf(self, response):
        try:
            doc = fitz.open(stream=response.body, filetype="pdf")
            text_blocks = []
            for page in doc:
                text_blocks.append(page.get_text())
            
            full_text = "\n".join(text_blocks).strip()
            
            if len(full_text) > 1000:
                yield {
                    "title": response.meta.get("title", "ArXiv Document"),
                    "text": full_text,
                    "url": response.meta.get("url"),
                    "language": "en", # ArXiv is mostly English, but we process it anyway
                    "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "char_count": len(full_text),
                    "word_count": len(full_text.split()),
                }
        except Exception as e:
            self.logger.error(f"Erro ao parsear PDF {response.url}: {e}")


class GutenbergPTSpider(scrapy.Spider):
    name = "gutenberg_pt"
    allowed_domains = ["gutenberg.org"]
    start_urls = ["https://www.gutenberg.org/browse/languages/pt"]

    def parse(self, response):
        # Extract book links from the language page
        # Example format: <li class="extiw"><a href="/ebooks/25641">A abelhinha</a>
        book_links = response.css("li.pgdbetext a::attr(href)").getall()
        for link in book_links:
            if link.startswith("/ebooks/"):
                book_id = link.split("/")[-1]
                # Gutenberg has plain text UTF-8 files under this predictable URL
                txt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                yield scrapy.Request(txt_url, callback=self.parse_book, meta={"url": response.urljoin(link)})

    def parse_book(self, response):
        text = response.text
        # Clean up Gutenberg headers/footers (basic cleanup)
        text = re.sub(r"^\*\*\* START OF THE PROJECT GUTENBERG.*?\*\*\*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\*\*\* END OF THE PROJECT GUTENBERG.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = text.strip()

        if len(text) > 1000:
            yield {
                "title": "Gutenberg PT Book",
                "text": text,
                "url": response.meta.get("url"),
                "language": "pt-br",
                "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "char_count": len(text),
                "word_count": len(text.split()),
            }


if __name__ == "__main__":
    max_docs = int(os.environ.get("MAX_DOCUMENTS", 100))
    output_bucket = os.environ.get("OUTPUT_BUCKET", "")
    spider_name = os.environ.get("SPIDER_NAME", "wikipedia_pt")
    
    settings = {
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 1.0,  # Atraso educado de 1s entre requisições
        "ITEM_PIPELINES": {
            "__main__.ParquetChunkPipeline": 300,
        },
        "CLOSESPIDER_ITEMCOUNT": max_docs,
        "LOG_LEVEL": "INFO",
    }
    
    process = CrawlerProcess(settings)
    
    if spider_name == "wikipedia_pt":
        process.crawl(WikipediaPTSpider)
    elif spider_name == "arxiv_pt":
        process.crawl(ArxivSpider)
    elif spider_name == "gutenberg_pt":
        process.crawl(GutenbergPTSpider)
    else:
        raise ValueError(f"Spider desconhecida: {spider_name}")
        
    process.start()
