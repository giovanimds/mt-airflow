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
        "https://pt.wikipedia.org/wiki/Especial:Aleat%C3%B3ria",
        "https://pt.wikipedia.org/wiki/Portal:Conte%C3%BAdo_destacado"
    ]

    def parse(self, response):
        # 1. Extrair conteúdo do artigo atual
        title = response.css("h1#firstHeading *::text").get()
        paragraphs = response.css("#mw-content-text .mw-parser-output p")
        
        # Limpar e juntar o texto dos parágrafos
        text_blocks = []
        for p in paragraphs:
            # Ignorar parágrafos que estejam dentro de infoboxes, tabelas, referências, etc.
            is_noise = p.xpath(
                "ancestor::*[contains(@class, 'infobox') or contains(@class, 'navbox') or contains(@class, 'metadata') or name()='table' or contains(@class, 'reflist')]"
            )
            if is_noise:
                continue
                
            text = "".join(p.css("*::text").getall()).strip()
            if text:
                # Remove citações de referências, ex: [1], [12], [nota 2]
                text = re.sub(r"\[(?:nota\s+)?\d+\]", "", text)
                text_blocks.append(text)
        
        full_text = "\n".join(text_blocks).strip()
        
        # Processar apenas páginas que contenham texto relevante
        if title and len(full_text) > 200:
            yield {
                "title": title.strip(),
                "text": full_text,
                "url": response.url,
                "language": "pt-br",
                "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
            }

        # 2. Seguir links para outras páginas da Wikipédia em português
        links = response.css("#mw-content-text a::attr(href)").getall()
        for link in links:
            # Ignorar namespaces especiais
            if link.startswith("/wiki/") and not re.search(
                r"^/wiki/(Wikip%C3%A9dia|Discuss%C3%A3o|Especial|Categoria|Ajuda|Ficheiro|Portal|Predefini%C3%A7%C3%A3o|MediaWiki|Anexo|Projeto):",
                link,
                re.IGNORECASE,
            ):
                if ":" not in link[6:]:
                    next_url = response.urljoin(link)
                    yield scrapy.Request(next_url, callback=self.parse)
                    
        # Yield ocasionalmente para a página aleatória para diversificar o rastreamento
        yield scrapy.Request(
            "https://pt.wikipedia.org/wiki/Especial:Aleat%C3%B3ria",
            callback=self.parse,
            priority=-1
        )


class RedditPTSpider(scrapy.Spider):
    name = "reddit_pt"
    allowed_domains = ["reddit.com"]
    start_urls = [
        "https://www.reddit.com/r/brasil/new.json?limit=100",
        "https://www.reddit.com/r/brasil/hot.json?limit=100",
        "https://www.reddit.com/r/brasil/top.json?limit=100&t=all",
        "https://www.reddit.com/r/portugal/new.json?limit=100",
        "https://www.reddit.com/r/portugal/hot.json?limit=100"
     ]
    
    custom_settings = {
        'USER_AGENT': 'bot:my_corpus_bot:v1.0 (by /u/corpus_builder)',
        'DOWNLOAD_DELAY': 1.5,
    }

    def parse(self, response):
        try:
            data = json.loads(response.text)
        except Exception as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            return

        if "data" not in data or "children" not in data["data"]:
            return

        for child in data["data"]["children"]:
            post = child.get("data", {})
            text = post.get("selftext", "").strip()
            
            # Só extrair se o post tiver texto considerável
            if len(text) > 200:
                yield {
                    "title": post.get("title", ""),
                    "text": text,
                    "url": f"https://www.reddit.com{post.get('permalink', '')}",
                    "language": "pt-br",
                    "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }
                
        after = data["data"].get("after")
        if after:
            base_url = response.url.split("?")[0]
            next_url = f"{base_url}?limit=100&after={after}"
            yield scrapy.Request(next_url, callback=self.parse)


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
    elif spider_name == "reddit_pt":
        process.crawl(RedditPTSpider)
    elif spider_name == "gutenberg_pt":
        process.crawl(GutenbergPTSpider)
    else:
        raise ValueError(f"Spider desconhecida: {spider_name}")
        
    process.start()
