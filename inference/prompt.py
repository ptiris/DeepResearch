TOOL_SCHEMAS = {
    "search": '{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}',
    "aliyun_search": '{"type": "function", "function": {"name": "aliyun_search", "description": "Perform web searches via Aliyun IQS then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}',
    "visit": '{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}',
    "PythonInterpreter": '{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:\n1. The \'arguments\' JSON object must be empty: {}.\n2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.\n\nIMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.\n\nExample of a correct call:\n<tool_call>\n{"name": "PythonInterpreter", "arguments": {}}\n<code>\nimport numpy as np\n# Your code here\nprint(f"The result is: {np.mean([1,2,3])}")\n</code>\n</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}',
    "google_scholar": '{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}, "required": ["query"]}}}',
    "parse_file": '{"type": "function", "function": {"name": "parse_file", "description": "This is a tool that can be used to parse multiple user uploaded local files such as PDF, DOCX, PPTX, TXT, CSV, XLSX, DOC, ZIP, MP4, MP3.", "parameters": {"type": "object", "properties": {"files": {"type": "array", "items": {"type": "string"}, "description": "The file name of the user uploaded local files to be parsed."}}, "required": ["files"]}}}',
}

SYSTEM_PROMPT_TEMPLATE = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schemas}
</tools>

## Important Format Rules

1. **Only ONE tool call per response**: Each response can contain at most ONE tool call.

2. **Tool call format**: Use <tool_call></tool_call> XML tags to wrap JSON object:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

3. **JSON requirements**:
   - Use standard JSON format with double quotes
   - Do NOT use single quotes
   - Do NOT nest XML tags inside JSON string values (e.g., {"name": "<search>...", "arguments": {}} is WRONG)

4. **Python Interpreter special format**: The code must be placed within <code></code> tags outside the JSON:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
# Your Python code here
print("Hello World")
</code>
</tool_call>

Current date: """

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace("{tool_schemas}", "\n".join(TOOL_SCHEMAS.values()))

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

REPHASE_PROMPT = """
You are given two search queries.

Merge them into ONE search query that:
- preserves ALL important information from both queries
- keeps constraints such as site:, year, location, names
- does NOT make the query more general
- is concise and suitable for a search engine

If merging would lose important information, return Query 1 unchanged.

Return ONLY the final query.

Query 1: {q1}
Query 2: {q2}
"""


def build_system_prompt(available_tools):
    """Build system prompt with only the specified available tools."""
    schemas = [TOOL_SCHEMAS[t] for t in available_tools if t in TOOL_SCHEMAS]
    tool_schemas_str = "\n".join(schemas)
    return SYSTEM_PROMPT_TEMPLATE.replace("{tool_schemas}", tool_schemas_str)
