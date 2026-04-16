import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import os


ALIYUN_IQS_API_KEY = os.environ.get('ALIYUN_IQS_API_KEY')
SERPER_KEY = os.environ.get('SERPER_KEY_ID')


@register_tool("aliyun_search", allow_overwrite=True)
class AliyunSearch(BaseTool):
    name = "aliyun_search"
    description = "Performs batched web searches via Aliyun IQS: supply an array 'query'; the tool retrieves the top search results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def search_with_aliyun(self, query: str):
        print(f"[AliyunSearch] Searching with Aliyun IQS for query: {query}")
        payload = {
            "query": query,
            "engineType": "Generic",
            "contents": {
                "mainText": False,
                "summary": True,
                "rerankScore": True
            }
        }
        headers = {
            "Authorization": f"Bearer {ALIYUN_IQS_API_KEY}",
            "Content-Type": "application/json"
        }
        print("Aliyun IQS Search Payload: ", json.dumps(payload))
        proxies = {}
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        print(f"Using proxies: {proxies}")
        last_status_code = None
        for i in range(5):
            try:
                res = requests.post(
                    "https://cloud-iqs.aliyuncs.com/search/unified",
                    json=payload,
                    headers=headers,
                    proxies=proxies,
                    timeout=30
                )
                last_status_code = res.status_code
                if res.status_code == 200:
                    results = res.json()
                    break
                else:
                    print(f"HTTP {res.status_code}: {res.text}")
                    if i == 4:
                        print("5 Attempts for Aliyun IQS search have failed")
                        return f"Aliyun IQS search failed with HTTP {res.status_code}, Please try again later.", last_status_code
                    continue
            except Exception as e:
                print(e)
                if i == 4:
                    print("5 Attempts for Aliyun IQS search have failed")
                    return f"Aliyun IQS search Timeout, return None, Please try again later.", None
                continue

        try:
            if "pageItems" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            for page in results["pageItems"]:
                idx += 1
                date_published = ""
                if "publishedTime" in page and page["publishedTime"]:
                    date_published = "\nDate published: " + page["publishedTime"]

                source = ""
                if "hostname" in page and page["hostname"]:
                    source = "\nSource: " + page["hostname"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                title = page.get("title", "No title")
                link = page.get("link", "")
                redacted_version = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
                web_snippets.append(redacted_version)

            content = f"An Aliyun IQS search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content, 200
        except:
            return f"No results found for '{query}'. Try with a more general query.", None

    def call(self, params: Union[str, dict], **kwargs) -> tuple:
        try:
            query = params["query"]
        except:
            return "[AliyunSearch] Invalid request format: Input must be a JSON object containing 'query' field", None

        if isinstance(query, str):
            response, status_code = self.search_with_aliyun(query)
        else:
            assert isinstance(query, List)
            responses = []
            status_codes = []
            for q in query:
                resp, sc = self.search_with_aliyun(q)
                responses.append(resp)
                if sc is not None:
                    status_codes.append(sc)
            response = "\n=======\n".join(responses)
            status_code = status_codes[0] if status_codes else (status_codes[-1] if status_codes else None)

        return response, status_code


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
    def google_search_with_serp(self, query: str):
        print(f"[Search] Searching with Serper for query: {query}")
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        if contains_chinese_basic(query):
            payload = {
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            }
        else:
            payload = {
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            }
        headers = {
                'X-API-KEY': SERPER_KEY,
                'Content-Type': 'application/json'
                }
        print("Search Payloads: ", json.dumps(payload))
        proxies = {}
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        print(f"Using proxies: {proxies}")
        last_status_code = None
        for i in range(5):
            try:
                res = requests.post("https://google.serper.dev/search", json=payload, headers=headers, proxies=proxies, timeout=30)
                last_status_code = res.status_code
                if res.status_code == 200:
                    results = res.json()
                    break
                else:
                    print(f"HTTP {res.status_code}: {res.text}")
                    if i == 4:
                        print("5 Attempts for search have failed")
                        return f"Google search failed with HTTP {res.status_code}, Please try again later.", last_status_code
                    continue
            except Exception as e:
                print(e)
                if i == 4:
                    print("5 Attempts for search have failed")
                    return f"Google search Timeout, return None, Please try again later.", None
                continue

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content, 200
        except:
            return f"No results found for '{query}'. Try with a more general query.", None


    
    def search_with_serp(self, query: str):
        result, status_code = self.google_search_with_serp(query)
        return result, status_code

    def call(self, params: Union[str, dict], **kwargs) -> tuple:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field", None
        
        if isinstance(query, str):
            response, status_code = self.search_with_serp(query)
        else:
            assert isinstance(query, List)
            responses = []
            status_codes = []
            for q in query:
                resp, sc = self.search_with_serp(q)
                responses.append(resp)
                if sc is not None:
                    status_codes.append(sc)
            response = "\n=======\n".join(responses)
            status_code = status_codes[0] if status_codes else (status_codes[-1] if status_codes else None)
            
        return response, status_code

