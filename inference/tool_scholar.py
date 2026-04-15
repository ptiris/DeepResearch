import os
import json
import requests
from typing import Union, List
from qwen_agent.tools.base import BaseTool, register_tool
from concurrent.futures import ThreadPoolExecutor


SERPER_KEY=os.environ.get('SERPER_KEY_ID')


@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"
    description = "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries."
    parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string", "description": "The search query."},
                    "minItems": 1,
                    "description": "The list of search queries for Google Scholar."
                },
            },
        "required": ["query"],
    }

    def google_scholar_with_serp(self, query: str):
        payload = {
            "q": query,
        }
        headers = {
            'X-API-KEY': SERPER_KEY,
            'Content-Type': 'application/json'
        }
        proxies = {}
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        last_status_code = None
        for i in range(5):
            try:
                res = requests.post("https://google.serper.dev/scholar", json=payload, headers=headers, proxies=proxies, timeout=30)
                last_status_code = res.status_code
                if res.status_code == 200:
                    results = res.json()
                    break
                else:
                    print(f"HTTP {res.status_code}: {res.text}")
                    if i == 4:
                        return f"Google Scholar failed with HTTP {res.status_code}, Please try again later.", last_status_code
                    continue
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google Scholar Timeout, return None, Please try again later.", None
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
                    if "year" in page:
                        date_published = "\nDate published: " + str(page["year"])

                    publicationInfo = ""
                    if "publicationInfo" in page:
                        publicationInfo = "\npublicationInfo: " + page["publicationInfo"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]
                    
                    link_info = "no available link"
                    if "pdfUrl" in page: 
                        link_info = "pdfUrl: " + page["pdfUrl"]
                    
                    citedBy = ""
                    if "citedBy" in page:
                        citedBy = "\ncitedBy: " + str(page["citedBy"])
                    
                    redacted_version = f"{idx}. [{page['title']}]({link_info}){publicationInfo}{date_published}{citedBy}\n{snippet}"

                    redacted_version = redacted_version.replace("Your browser can't play this video.", "") 
                    web_snippets.append(redacted_version)

            content = f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
            return content, 200
        except:
            return f"No results found for '{query}'. Try with a more general query.", None


    def call(self, params: Union[str, dict], **kwargs) -> tuple:
        try:
            params = self._verify_json_format_args(params)
            query = params["query"]
        except:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field", None
        
        if isinstance(query, str):
            response, status_code = self.google_scholar_with_serp(query)
        else:
            assert isinstance(query, List)
            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(self.google_scholar_with_serp, query))
            responses = [r[0] for r in results]
            status_codes = [r[1] for r in results if r[1] is not None]
            response = "\n=======\n".join(responses)
            status_code = status_codes[0] if status_codes else (status_codes[-1] if status_codes else None)
        return response, status_code
