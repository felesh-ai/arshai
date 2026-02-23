Multimodal Support — Images and PDFs
=====================================

Arshai supports images and PDF documents natively across all LLM providers.
Pass base64-encoded content via ``images_base64`` and ``pdfs_base64`` on
``ILLMInput`` — the framework handles provider-specific wire formats
automatically.

Philosophy
----------

- **Base64-only**: One universal format accepted by every provider.
- **Developer-controlled**: The framework passes content as-is. You choose image processing libraries, caching strategies, and optimisations.
- **No hidden parsing**: PDFs are forwarded natively to the model. The framework never extracts text server-side.
- **Caller picks the model**: It is your responsibility to choose a model that supports the required modality.

Images
------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import base64
   from arshai.core.interfaces.illm import ILLMInput, ILLMConfig
   from arshai.llms.openai import OpenAIClient

   config = ILLMConfig(model="gpt-4o", temperature=0.2)
   client = OpenAIClient(config)

   with open("photo.jpg", "rb") as f:
       img_b64 = base64.b64encode(f.read()).decode("utf-8")

   input_data = ILLMInput(
       system_prompt="You are a vision assistant.",
       user_message="Describe what you see in this image.",
       images_base64=[img_b64]
   )

   response = await client.chat(input_data)
   print(response["llm_response"])

Both raw base64 strings and ``data:image/*;base64,...`` data URLs are accepted.
You can mix formats freely within the same list.

Multiple Images
~~~~~~~~~~~~~~~

.. code-block:: python

   images = []
   for path in ["before.jpg", "after.jpg"]:
       with open(path, "rb") as f:
           images.append(base64.b64encode(f.read()).decode("utf-8"))

   input_data = ILLMInput(
       system_prompt="You are a visual comparison expert.",
       user_message="What changed between these two images?",
       images_base64=images
   )

Optional Helper
~~~~~~~~~~~~~~~

.. code-block:: python

   from arshai.llms.utils.images import image_file_to_base64, image_url_to_base64

   img = image_file_to_base64("photo.jpg")
   img = image_url_to_base64("https://example.com/photo.jpg")

For production use, implement your own helpers with resizing, compression, caching, and async loading.

Size Limits
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Provider
     - Limit
     - Recommendation
   * - Gemini
     - 20 MB inline
     - Resize to 2048×2048 max
   * - OpenAI
     - ~4 MB
     - Compress to JPEG 85% quality
   * - Azure
     - ~4 MB
     - Same as OpenAI
   * - OpenRouter
     - Varies by model
     - Check model documentation

PDFs
----

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import base64
   from arshai.core.interfaces.illm import ILLMInput

   with open("report.pdf", "rb") as f:
       pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

   input_data = ILLMInput(
       system_prompt="You are a document analyst.",
       user_message="Summarise this PDF and list the three most important findings.",
       pdfs_base64=[pdf_b64]
   )

   response = await client.chat(input_data)
   print(response["llm_response"])

Both raw base64 strings and ``data:application/pdf;base64,...`` data URLs are accepted.

Multiple PDFs
~~~~~~~~~~~~~

.. code-block:: python

   pdfs = []
   for path in ["q1_report.pdf", "q2_report.pdf"]:
       with open(path, "rb") as f:
           pdfs.append(base64.b64encode(f.read()).decode("utf-8"))

   input_data = ILLMInput(
       system_prompt="You are a financial analyst.",
       user_message="Compare Q1 and Q2 and identify the biggest change.",
       pdfs_base64=pdfs
   )

Optional Helper
~~~~~~~~~~~~~~~

.. code-block:: python

   from arshai.llms.utils.pdfs import pdf_file_to_base64, pdf_url_to_base64

   pdf = pdf_file_to_base64("report.pdf")
   pdf = pdf_url_to_base64("https://example.com/report.pdf")

PDF + Tools
~~~~~~~~~~~

PDFs work alongside regular function calling:

.. code-block:: python

   def record_finding(finding: str, page: int = 0) -> str:
       """Record a key finding extracted from the document."""
       print(f"Page {page}: {finding}")
       return "Finding recorded."

   input_data = ILLMInput(
       system_prompt="Read the PDF and record each key finding using record_finding.",
       user_message="Extract all findings from this document.",
       pdfs_base64=[pdf_b64],
       regular_functions={"record_finding": record_finding},
       max_turns=10
   )

Images and PDFs Together
------------------------

.. code-block:: python

   input_data = ILLMInput(
       system_prompt="You are a research assistant. Analyse all provided content.",
       user_message="Describe the chart and relate it to the figures in the PDF.",
       images_base64=[chart_img_b64],
       pdfs_base64=[annual_report_pdf_b64]
   )

Provider Wire Formats
---------------------

The framework applies the correct format for each provider automatically.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Provider
     - PDF format
     - Notes
   * - Gemini
     - ``Part.from_bytes(mime_type="application/pdf")``
     - Native SDK, bytes decoded from base64
   * - OpenAI
     - ``input_file`` block (Responses API)
     - Direct API with data URL
   * - Azure
     - ``input_file`` block (Responses API)
     - Same as OpenAI
   * - OpenRouter
     - ``file`` block + ``plugins: native``
     - Forwarded natively, no server-side parsing
   * - AI Gateway
     - ``file`` block (Chat Completions)
     - Works with any OpenAI-compatible gateway
   * - Cloudflare Gateway
     - ``file`` block (Chat Completions)
     - Delegates to AI Gateway internally

Streaming with Multimodal Input
--------------------------------

Images and PDFs work with both ``chat()`` and ``stream()``:

.. code-block:: python

   async for chunk in client.stream(input_data):
       if chunk.get("llm_response"):
           print(chunk["llm_response"], end="", flush=True)

Images and PDFs are sent in the initial request. The model retains context
across subsequent function-calling turns — content is not re-sent on every turn.

Backward Compatibility
----------------------

Both ``images_base64`` and ``pdfs_base64`` default to ``[]``. Existing code
that does not use them is completely unaffected:

.. code-block:: python

   # Unchanged — works exactly as before
   input_data = ILLMInput(
       system_prompt="You are a helpful assistant.",
       user_message="Hello!"
   )
   assert input_data.images_base64 == []
   assert input_data.pdfs_base64 == []
