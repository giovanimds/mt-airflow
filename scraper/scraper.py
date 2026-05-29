from __future__ import annotations

import datetime
import json
import os
import re
import uuid
import sys
import subprocess
import time
import pandas as pd
import scrapy
from itemadapter import ItemAdapter
from scrapy.crawler import CrawlerProcess
import fitz  # PyMuPDF
from langdetect import detect, DetectorFactory
import redis
import psycopg2

DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    try:
        if len(text.strip()) < 10:
            return "unknown"
        return detect(text)
    except Exception:
        return "unknown"

def get_search_topic(redis_url: str) -> str | None:
    try:
        r = redis.Redis.from_url(redis_url, socket_timeout=5)
        topic_bytes = r.rpop("search_topics_queue")
        if topic_bytes:
            topic = topic_bytes.decode("utf-8")
            print(f"Obtido tema de busca do Valkey: {topic}")
            return topic
    except Exception as e:
        print(f"Erro ao obter tema de busca do Valkey: {e}")
    return None


class YugabyteDBCorpusPipeline:
    def __init__(self, redis_url: str, db_host: str, db_port: int, db_user: str, db_pass: str, db_name: str, chunk_size: int = 50):
        self.redis_url = redis_url
        self.db_host = db_host
        self.db_port = db_port
        self.db_user = db_user
        self.db_pass = db_pass
        self.db_name = db_name
        self.chunk_size = chunk_size
        self.items = []
        self.db_conn = None
        self.redis_client = None

    @classmethod
    def from_crawler(cls, crawler):
        redis_url = os.environ.get("REDIS_URL") or crawler.settings.get("REDIS_URL", "redis://valkey-primary.default.svc.cluster.local:6379")
        db_host = os.environ.get("PG_HOST") or crawler.settings.get("PG_HOST", "postgres.morescotech.com.br")
        db_port = int(os.environ.get("PG_PORT") or crawler.settings.get("PG_PORT", 5432))
        db_user = os.environ.get("PG_USER") or crawler.settings.get("PG_USER", "yugabyte")
        db_pass = os.environ.get("PG_PASSWORD") or crawler.settings.get("PG_PASSWORD", "YugabytePass2026")
        db_name = os.environ.get("PG_DATABASE") or crawler.settings.get("PG_DATABASE", "ai_labs")
        chunk_size = int(os.environ.get("CHUNK_SIZE", 50))
        return cls(redis_url, db_host, db_port, db_user, db_pass, db_name, chunk_size)

    def open_spider(self, spider):
        spider.logger.info(f"Conectando ao YugabyteDB em {self.db_host}:{self.db_port}/{self.db_name} e Valkey em {self.redis_url}")
        params = {
            "host": self.db_host,
            "port": self.db_port,
            "user": self.db_user,
            "password": self.db_pass,
            "database": self.db_name,
            "sslmode": "disable"
        }
        try:
            self.db_conn = psycopg2.connect(**params, load_balance=True)
        except TypeError:
            self.db_conn = psycopg2.connect(**params)
        self.redis_client = redis.Redis.from_url(self.redis_url)

    def close_spider(self, spider):
        if self.items:
            self.write_batch(spider)
        if self.db_conn:
            self.db_conn.close()
            spider.logger.info("Conexão com YugabyteDB encerrada.")

    def process_item(self, item, spider):
        self.items.append(dict(item))
        if len(self.items) >= self.chunk_size:
            self.write_batch(spider)
        return item

    def write_batch(self, spider):
        from psycopg2.extras import execute_values
        batch = self.items
        self.items = []
        
        spider.logger.info(f"Inserindo batch de {len(batch)} itens no YugabyteDB...")
        
        def clean_val(v):
            if v is None:
                return None
            if isinstance(v, str):
                return v.replace('\x00', '').replace('\u0000', '')
            return str(v).replace('\x00', '').replace('\u0000', '')

        rows_to_insert = []
        for row in batch:
            title = clean_val(row.get("title", ""))
            text = clean_val(row.get("text", ""))
            url = clean_val(row.get("url", ""))
            language = clean_val(row.get("language", "pt"))
            extracted_at = row.get("extracted_at")
            char_count = row.get("char_count", len(text))
            word_count = row.get("word_count", len(text.split()))
            
            rows_to_insert.append((
                title, text, url, language, spider.name, extracted_at, char_count, word_count
            ))

        try:
            with self.db_conn.cursor() as cur:
                query = """
                INSERT INTO raw_corpus (title, text, url, language, spider_name, extracted_at, char_count, word_count)
                VALUES %s
                ON CONFLICT (url) DO NOTHING
                RETURNING id;
                """
                inserted_ids = execute_values(
                    cur,
                    query,
                    rows_to_insert,
                    template=None,
                    page_size=100,
                    fetch=True
                )
            self.db_conn.commit()
            
            new_ids = [r[0] for r in inserted_ids] if inserted_ids else []
            spider.logger.info(f"Batch concluído: {len(new_ids)} novos registros de {len(batch)} inseridos.")
            
            if new_ids:
                spider.logger.info(f"Enfileirando {len(new_ids)} novos IDs no Valkey (raw_corpus_queue)...")
                for nid in new_ids:
                    self.redis_client.lpush("raw_corpus_queue", json.dumps({"id": str(nid)}))
                    
        except Exception as e:
            spider.logger.error(f"Erro ao inserir batch no banco: {e}")
            self.db_conn.rollback()


class WikipediaPTSpider(scrapy.Spider):
    name = "wikipedia_pt"
    allowed_domains = ["pt.wikipedia.org"]

    def start_requests(self):
        redis_url = self.settings.get("REDIS_URL") or os.environ.get("REDIS_URL", "redis://valkey-primary.default.svc.cluster.local:6379")
        topic = get_search_topic(redis_url)
        if topic:
            self.logger.info(f"Pesquisando na Wikipedia por tema guiado: '{topic}'")
            search_url = f"https://pt.wikipedia.org/w/api.php?action=opensearch&search={topic}&limit=100&format=json"
            yield scrapy.Request(search_url, callback=self.parse_search)
        else:
            self.logger.info("Nenhum tema na fila. Usando busca aleatória padrão do Wikipedia.")
            start_url = "https://pt.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0&prop=extracts&explaintext=1&format=json&grnlimit=1"
            yield scrapy.Request(start_url, callback=self.parse)

    def parse_search(self, response):
        try:
            data = json.loads(response.text)
            titles = data[1]
            self.logger.info(f"Busca retornou {len(titles)} páginas da Wikipedia.")
            for title in titles:
                page_url = f"https://pt.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&explaintext=1&format=json"
                yield scrapy.Request(page_url, callback=self.parse_page)
        except Exception as e:
            self.logger.error(f"Erro ao parsear busca Wikipedia: {e}")

    def parse_page(self, response):
        try:
            data = json.loads(response.text)
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                title = page_info.get("title", "")
                text = page_info.get("extract", "").strip()

                if title and len(text) > 200:
                    yield {
                        "title": title,
                        "text": text,
                        "url": f"https://pt.wikipedia.org/wiki/?curid={page_id}",
                        "language": detect_language(text),
                        "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                        "char_count": len(text),
                        "word_count": len(text.split()),
                    }
        except Exception as e:
            self.logger.error(f"Erro ao parsear página Wikipedia: {e}")

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
                    "language": detect_language(text),
                    "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }

        # Continuar solicitando páginas aleatórias infinitamente (limitado pelo CLOSESPIDER_ITEMCOUNT global)
        yield scrapy.Request(
            self.start_urls[0] if hasattr(self, 'start_urls') else "https://pt.wikipedia.org/w/api.php?action=query&generator=random&grnnamespace=0&prop=extracts&explaintext=1&format=json&grnlimit=1",
            callback=self.parse,
            dont_filter=True
        )


class ArxivSpider(scrapy.Spider):
    name = "arxiv_pt"
    allowed_domains = ["export.arxiv.org", "arxiv.org"]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 3.0,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 3.0,
        'AUTOTHROTTLE_MAX_DELAY': 60.0,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 1.0,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'RETRY_HTTP_CODES': [429, 500, 502, 503, 504, 522, 524, 408],
        'RETRY_TIMES': 15,
        'USER_AGENT': 'Mozilla/5.0 (compatible; mt-airflow-scraper/1.0; +http://example.com)'
    }

    def start_requests(self):
        redis_url = self.settings.get("REDIS_URL") or os.environ.get("REDIS_URL", "redis://valkey-primary.default.svc.cluster.local:6379")
        topic = get_search_topic(redis_url)
        if topic:
            self.logger.info(f"Pesquisando no ArXiv por tema guiado: '{topic}'")
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic)
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_topic}&start=0&max_results=100"
            yield scrapy.Request(url, callback=self.parse_api, meta={"start": 0, "topic": topic})
        else:
            self.logger.info("Nenhum tema na fila. Usando busca padrão cat:cs.AI no ArXiv.")
            url = "http://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=1000"
            yield scrapy.Request(url, callback=self.parse_api, meta={"start": 0})

    def parse_api(self, response):
        response.selector.remove_namespaces()
        entries = response.css("entry")
        
        if not entries:
            self.logger.warning("No more entries found or rate limit hit.")
            return

        for entry in entries:
            title = entry.css("title::text").get(default="").strip()
            pdf_url = entry.css("link[type='application/pdf']::attr(href)").get()
            
            if pdf_url:
                yield scrapy.Request(
                    pdf_url,
                    callback=self.parse_pdf,
                    meta={"title": title, "url": pdf_url}
                )
                
        # Next page (apenas se não for busca guiada)
        if "topic" not in response.meta:
            start = response.meta["start"] + 1000
            next_url = f"http://export.arxiv.org/api/query?search_query=cat:cs.AI&start={start}&max_results=1000"
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
                    "language": detect_language(full_text),
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
        book_links = response.css("li.pgdbetext a::attr(href)").getall()
        for link in book_links:
            if link.startswith("/ebooks/"):
                book_id = link.split("/")[-1]
                txt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                yield scrapy.Request(txt_url, callback=self.parse_book, meta={"url": response.urljoin(link)})

    def parse_book(self, response):
        text = response.text
        text = re.sub(r"^\*\*\* START OF THE PROJECT GUTENBERG.*?\*\*\*", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"\*\*\* END OF THE PROJECT GUTENBERG.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = text.strip()

        if len(text) > 1000:
            yield {
                "title": "Gutenberg PT Book",
                "text": text,
                "url": response.meta.get("url"),
                "language": detect_language(text),
                "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "char_count": len(text),
                "word_count": len(text.split()),
            }


class SciELOSpider(scrapy.Spider):
    name = "scielo_pt"
    allowed_domains = ["scielo.br"]
    start_urls = ["https://www.scielo.br/journals/alpha?status=current"]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1.0,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'RETRY_HTTP_CODES': [403, 429, 500, 502, 503, 504],
        'RETRY_TIMES': 5,
    }

    def parse(self, response):
        journal_links = response.xpath("//a[contains(@href, '/j/')]/@href").getall()
        for link in set(journal_links):
            if not link.endswith("/"):
                link += "/"
            yield scrapy.Request(response.urljoin(link + "grid"), callback=self.parse_journal_grid)

    def parse_journal_grid(self, response):
        issue_links = response.xpath("//a[contains(@href, '/i/')]/@href").getall()
        for link in set(issue_links):
            yield scrapy.Request(response.urljoin(link), callback=self.parse_issue)

    def parse_issue(self, response):
        article_links = response.xpath("//a[contains(@href, '/a/')]/@href").getall()
        for link in set(article_links):
            yield scrapy.Request(response.urljoin(link), callback=self.parse_article)

    def parse_article(self, response):
        texts = response.xpath("//div[contains(@class, 'articleSection')]//p//text() | //div[contains(@class, 'content')]//p//text() | //div[@class='html-body']//p//text() | //body//p//text()").getall()
        full_text = " ".join([t.strip() for t in texts if t.strip()])
        title = response.xpath("//h1[@class='article-title']/text() | //h1/text()").get(default="SciELO Article").strip()
        
        if len(full_text) > 500:
            yield {
                "title": title,
                "text": full_text,
                "url": response.url,
                "language": detect_language(full_text),
                "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
            }


class BdtdSpider(scrapy.Spider):
    name = "bdtd_pt"

    def start_requests(self):
        redis_url = self.settings.get("REDIS_URL") or os.environ.get("REDIS_URL", "redis://valkey-primary.default.svc.cluster.local:6379")
        topic = get_search_topic(redis_url)
        if topic:
            self.logger.info(f"Pesquisando no BDTD por tema guiado: '{topic}'")
            import urllib.parse
            encoded_topic = urllib.parse.quote(topic)
            url = f"https://bdtd.ibict.br/vufind/Search/Results?lookfor={encoded_topic}&type=AllFields"
            yield scrapy.Request(url, callback=self.parse, meta={"guided": True})
        else:
            self.logger.info("Nenhum tema na fila. Usando busca padrão 'ciencia matematica' no BDTD.")
            url = "https://bdtd.ibict.br/vufind/Search/Results?lookfor=ciencia+matematica&type=AllFields"
            yield scrapy.Request(url, callback=self.parse, meta={"guided": False})

    def parse(self, response):
        record_links = response.xpath("//a[contains(@href, '/vufind/Record/')]/@href").getall()
        for link in set(record_links):
            yield scrapy.Request(response.urljoin(link), callback=self.parse_record)
            
        next_page = response.xpath("//a[contains(@class, 'page-link') and contains(@aria-label, 'Next')]/@href | //a[contains(@class, 'next')]/@href | //a[@title='Próxima página']/@href").get()
        if next_page:
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse, meta=response.meta)

    def parse_record(self, response):
        external_links = response.xpath("//a[contains(@class, 'btn-primary') and contains(@href, 'http')]/@href | //table//a[contains(@href, 'http')]/@href").getall()
        for link in set(external_links):
            if "ibict.br" not in link:
                yield scrapy.Request(link, callback=self.parse_university_page)

    def parse_university_page(self, response):
        pdf_links = response.xpath("//a[contains(@href, '.pdf')]/@href | //a[contains(@href, 'bitstream')]/@href").getall()
        for link in set(pdf_links):
            yield scrapy.Request(response.urljoin(link), callback=self.parse_pdf)

    def parse_pdf(self, response):
        try:
            doc = fitz.open(stream=response.body, filetype="pdf")
            text_blocks = [page.get_text() for page in doc]
            full_text = "\n".join(text_blocks).strip()
            
            if len(full_text) > 1000:
                yield {
                    "title": "BDTD Thesis/Dissertation",
                    "text": full_text,
                    "url": response.url,
                    "language": detect_language(full_text),
                    "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "char_count": len(full_text),
                    "word_count": len(full_text.split()),
                }
        except Exception as e:
            self.logger.error(f"Erro ao parsear PDF do BDTD {response.url}: {e}")


class BolemaSpider(scrapy.Spider):
    name = "bolema_pt"
    start_urls = ["https://www.scielo.br/j/bolema/grid"]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1.0,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    def parse(self, response):
        issue_links = response.xpath("//a[contains(@href, '/j/bolema/i/')]/@href").getall()
        for link in issue_links:
            yield scrapy.Request(response.urljoin(link), callback=self.parse_issue)

    def parse_issue(self, response):
        article_links = response.xpath("//a[contains(@href, '/j/bolema/a/')]/@href").getall()
        for link in article_links:
            yield scrapy.Request(response.urljoin(link), callback=self.parse_article)

    def parse_article(self, response):
        texts = response.xpath("//div[contains(@class, 'articleSection')]//p//text() | //div[contains(@class, 'content')]//p//text() | //div[@class='html-body']//p//text() | //body//p//text()").getall()
        full_text = " ".join([t.strip() for t in texts if t.strip()])
        title = response.xpath("//h1[@class='article-title']/text() | //h1/text()").get(default="Bolema Article").strip()
        
        if len(full_text) > 500:
            yield {
                "title": title,
                "text": full_text,
                "url": response.url,
                "language": detect_language(full_text),
                "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
            }


class RematSpider(scrapy.Spider):
    name = "remat_pt"
    start_urls = ["https://periodicos.ifrs.edu.br/index.php/REMAT/issue/archive"]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 2.0,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    }

    def parse(self, response):
        issue_links = response.xpath("//a[contains(@class, 'title')]/@href").getall()
        for link in issue_links:
            yield scrapy.Request(response.urljoin(link), callback=self.parse_issue)
            
        next_page = response.xpath("//a[@class='next']/@href").get()
        if next_page:
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse)

    def parse_issue(self, response):
        article_links = response.xpath("//div[@class='title']/a/@href").getall()
        for link in article_links:
            yield scrapy.Request(response.urljoin(link), callback=self.parse_article)

    def parse_article(self, response):
        pdf_link = response.xpath("//a[contains(@class, 'galley-link') and contains(@class, 'pdf')]/@href").get()
        if pdf_link:
            pdf_url = response.urljoin(pdf_link)
            yield scrapy.Request(pdf_url, callback=self.parse_pdf)

    def parse_pdf(self, response):
        real_pdf_link = response.xpath("//a[contains(@class, 'download')]/@href").get()
        if real_pdf_link and not response.body.startswith(b'%PDF'):
            yield scrapy.Request(response.urljoin(real_pdf_link), callback=self.parse_pdf)
            return

        try:
            doc = fitz.open(stream=response.body, filetype="pdf")
            text_blocks = [page.get_text() for page in doc]
            full_text = "\n".join(text_blocks).strip()
            
            if len(full_text) > 500:
                yield {
                    "title": "REMAT Article",
                    "text": full_text,
                    "url": response.url,
                    "language": detect_language(full_text),
                    "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "char_count": len(full_text),
                    "word_count": len(full_text.split()),
                }
        except Exception as e:
            self.logger.error(f"Erro ao parsear PDF da REMAT {response.url}: {e}")


def run_crawler_subprocess():
    max_docs = int(os.environ.get("MAX_DOCUMENTS", 100))
    spider_name = os.environ.get("SPIDER_NAME", "wikipedia_pt")
    
    settings = {
        "ROBOTSTXT_OBEY": False,
        "CONCURRENT_REQUESTS": 8,
        "DOWNLOAD_DELAY": 1.0,
        "ITEM_PIPELINES": {
            "__main__.YugabyteDBCorpusPipeline": 300,
        },
        "CLOSESPIDER_ITEMCOUNT": max_docs,
        "LOG_LEVEL": "INFO",
        
        # Redis & Bloom Filter Settings
        "DUPEFILTER_CLASS": "scrapy_redis_bloomfilter.dupefilter.RFPDupeFilter",
        "SCHEDULER": "scrapy_redis_bloomfilter.scheduler.Scheduler",
        "SCHEDULER_PERSIST": True,
        "REDIS_URL": os.environ.get("REDIS_URL", "redis://valkey-service.default.svc.cluster.local:6379"),
        "BLOOMFILTER_HASH_NUMBER": 6,
        "BLOOMFILTER_BIT": 24, # 2^24 bits = ~2MB
    }
    
    process = CrawlerProcess(settings)
    
    if spider_name == "wikipedia_pt":
        process.crawl(WikipediaPTSpider)
    elif spider_name == "arxiv_pt":
        process.crawl(ArxivSpider)
    elif spider_name == "gutenberg_pt":
        process.crawl(GutenbergPTSpider)
    elif spider_name == "scielo_pt":
        process.crawl(SciELOSpider)
    elif spider_name == "bdtd_pt":
        process.crawl(BdtdSpider)
    elif spider_name == "bolema_pt":
        process.crawl(BolemaSpider)
    elif spider_name == "remat_pt":
        process.crawl(RematSpider)
    else:
        raise ValueError(f"Spider desconhecida: {spider_name}")
        
    process.start()

if __name__ == "__main__":
    if os.environ.get("RUN_SPIDER_SUBPROCESS") == "1":
        run_crawler_subprocess()
    else:
        print("Iniciando loop do Scraper Daemon...")
        redis_url = os.environ.get("REDIS_URL", "redis://valkey-service.default.svc.cluster.local:6379")
        r = redis.Redis.from_url(redis_url)
        
        while True:
            try:
                # Checar se há temas pendentes na fila
                queue_len = r.llen("search_topics_queue")
                if queue_len == 0:
                    # Dorme um pouco para não onerar o CPU
                    time.sleep(10)
                    continue
                
                print(f"Encontrados {queue_len} temas de busca na fila. Disparando subprocesso spider...")
                env = os.environ.copy()
                env["RUN_SPIDER_SUBPROCESS"] = "1"
                
                # Executa o spider em um subprocesso (para isolar o reactor do Twisted que não reinicia)
                subprocess.run([sys.executable, __file__], env=env)
                print("Spider finalizado. Aguardando 2s antes do próximo ciclo...")
                time.sleep(2)
                
            except redis.exceptions.ConnectionError:
                print("Conexão perdida com Valkey no Scraper Daemon. Tentando reconectar...")
                time.sleep(5)
            except Exception as e:
                print(f"Erro no loop do Scraper Daemon: {e}")
                time.sleep(10)
