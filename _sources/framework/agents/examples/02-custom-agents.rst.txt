Example 02: Creating Custom Agents
===================================

This example demonstrates how to create specialized agents for specific tasks, showing different return types and custom processing logic.

**File**: ``examples/agents/02_custom_agents.py``

**Prerequisites**: Set ``OPENROUTER_API_KEY`` environment variable

Overview
--------

This example shows three specialized agents:

- **SentimentAnalysisAgent**: Returns structured sentiment analysis with confidence scores
- **TranslationAgent**: Translates text with alternative translations and cultural notes  
- **CodeReviewAgent**: Reviews code with structured feedback and recommendations

Each agent demonstrates different patterns for specialized functionality and custom return types.

Key Concepts Demonstrated
-------------------------

**Structured Output Agents**

Unlike the basic example that returns strings, these agents return structured data:

.. code-block:: python

   class SentimentAnalysisAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Returns structured dictionary instead of string
           return {
               "sentiment": "positive/negative/neutral",
               "confidence": 85,
               "indicators": ["happy", "excited"],
               "explanation": "The text shows positive emotions..."
           }

**Custom System Prompts**

Each agent has a specialized system prompt tailored to its function:

.. code-block:: python

   # Sentiment analysis prompt
   system_prompt = """You are a sentiment analysis expert. 
   Analyze the emotional tone of messages and provide:
   1. Overall sentiment (positive/negative/neutral)
   2. Confidence score (0-100%)
   3. Key emotional indicators
   
   Always respond in this JSON format:
   {
       "sentiment": "positive/negative/neutral",
       "confidence": 0-100,
       "indicators": ["list", "of", "indicators"],
       "explanation": "brief explanation"
   }"""

**Error Handling and Fallbacks**

Agents include robust error handling for parsing and processing:

.. code-block:: python

   try:
       analysis = json.loads(response_text)
   except json.JSONDecodeError:
       # Fallback if parsing fails
       analysis = {
           "sentiment": "unknown",
           "confidence": 0,
           "indicators": ["parsing_error"],
           "explanation": "Failed to parse LLM response"
       }

Agent Implementations
---------------------

**1. SentimentAnalysisAgent**

Specializes in emotional tone analysis with structured output:

**Capabilities**:
- Analyzes emotional tone of text
- Provides confidence scores (0-100%)
- Identifies key emotional indicators
- Maintains analysis history for summary statistics

**Key Features**:

.. code-block:: python

   # Tracks analysis history
   self.analysis_history = []
   
   # Provides summary statistics
   def get_summary(self) -> Dict[str, Any]:
       return {
           "total_analyses": len(self.analysis_history),
           "average_confidence": avg_confidence,
           "sentiment_distribution": {
               "positive": positive_count,
               "negative": negative_count,
               "neutral": neutral_count
           }
       }

**Example Usage**:

.. code-block:: python

   sentiment_agent = SentimentAnalysisAgent(llm_client)
   
   analysis = await sentiment_agent.process(IAgentInput(
       message="I absolutely love this new feature! It's amazing!"
   ))
   
   # Returns structured analysis
   print(f"Sentiment: {analysis['sentiment']}")
   print(f"Confidence: {analysis['confidence']}%")
   print(f"Indicators: {', '.join(analysis['indicators'])}")

**2. TranslationAgent**

Specializes in language translation with cultural context:

**Capabilities**:
- Translates between multiple languages
- Preserves tone and context
- Provides alternative translations
- Includes cultural notes when relevant

**Key Features**:

.. code-block:: python

   # Dynamic target language via metadata
   target_lang = input.metadata.get("target_language", self.target_language)
   
   # Returns comprehensive translation data
   return {
       "translation": "main translation",
       "alternatives": ["alternative1", "alternative2"],
       "notes": "cultural context notes",
       "source_text": original_text,
       "target_language": target_lang
   }

**Example Usage**:

.. code-block:: python

   translator = TranslationAgent(llm_client, target_language="French")
   
   # Basic translation
   result = await translator.process(IAgentInput(
       message="Hello, how are you today?"
   ))
   
   # Override target language with metadata
   result = await translator.process(IAgentInput(
       message="Good morning!",
       metadata={"target_language": "Japanese"}
   ))

**3. CodeReviewAgent**

Specializes in code analysis and review:

**Capabilities**:
- Reviews code for best practices and conventions
- Identifies potential bugs and issues
- Provides performance and security analysis
- Suggests improvements and highlights positive aspects

**Key Features**:

.. code-block:: python

   # Language-specific expertise
   def __init__(self, llm_client: ILLM, language: str = "Python"):
       self.language = language
       
   # Structured review format
   return {
       "overall_quality": "excellent/good/fair/needs_improvement",
       "issues": [
           {
               "type": "bug/style/performance/security",
               "line": line_number,
               "description": "issue description",
               "severity": "high/medium/low"
           }
       ],
       "suggestions": ["improvement suggestions"],
       "positive_aspects": ["what was done well"]
   }

**Example Usage**:

.. code-block:: python

   code_reviewer = CodeReviewAgent(llm_client, language="Python")
   
   sample_code = """
   def calculate_average(numbers):
       sum = 0
       for i in range(len(numbers)):
           sum = sum + numbers[i]
       average = sum / len(numbers)
       return average
   """
   
   review = await code_reviewer.process(IAgentInput(message=sample_code))
   print(f"Quality: {review['overall_quality']}")
   print(f"Issues: {len(review['issues'])}")

Advanced Patterns
-----------------

**1. JSON Response Parsing**

All agents demonstrate robust JSON parsing with fallbacks:

.. code-block:: python

   try:
       structured_data = json.loads(response_text)
       # Enhance with additional metadata
       structured_data["source_text"] = input.message
       structured_data["processing_time"] = timestamp
   except json.JSONDecodeError:
       # Provide fallback structure
       structured_data = {
           "error": "parsing_failed",
           "raw_response": response_text,
           "source_text": input.message
       }

**2. Metadata-Driven Behavior**

Agents can modify behavior based on input metadata:

.. code-block:: python

   # Check for behavior overrides in metadata
   target_lang = input.metadata.get("target_language", self.target_language)
   max_length = input.metadata.get("max_length", None)
   analysis_depth = input.metadata.get("depth", "standard")
   
   # Adapt processing accordingly
   if analysis_depth == "deep":
       # Use more detailed analysis
       prompt = self.deep_analysis_prompt
   else:
       prompt = self.standard_prompt

**3. State Tracking (Instance Variables)**

Agents can maintain state for analytics without violating stateless principles:

.. code-block:: python

   # Track operations for summary (not conversation state)
   self.analysis_history.append({
       "input": input.message,
       "analysis": analysis,
       "timestamp": datetime.now()
   })
   
   # Provide operational insights
   def get_summary(self) -> Dict[str, Any]:
       # Returns operational statistics, not conversational state
       return analytics_data

Running the Example
-------------------

**Setup**:

.. code-block:: bash

   export OPENROUTER_API_KEY=your_key_here
   cd examples/agents
   python 02_custom_agents.py

**Expected Output**:

.. code-block:: text

   ============================================================
   SENTIMENT ANALYSIS AGENT
   ============================================================
   
   📝 Text: I absolutely love this new feature! It's amazing and works perfectly!
   📊 Analysis:
      Sentiment: positive
      Confidence: 95%
      Indicators: love, amazing, perfectly
      Explanation: The text shows strong positive emotions...
   
   📈 Summary: {'total_analyses': 3, 'average_confidence': 83.33, ...}
   
   ============================================================
   TRANSLATION AGENT
   ============================================================
   
   🌐 Original: Hello, how are you today?
   🇫🇷 Translation: Bonjour, comment allez-vous aujourd'hui ?
      Alternatives: Salut, comment ça va ?, Hello, comment vous portez-vous ?
   
   🔄 Using metadata to override target language...
   🇯🇵 Japanese: こんにちは！元気ですか？
   
   ============================================================
   CODE REVIEW AGENT
   ============================================================
   
   🔍 Code Review Results:
      Overall Quality: needs_improvement
      Issues Found: 3
         - [medium] style: Using 'sum' as variable name shadows built-in
         - [low] performance: Consider using sum() built-in function
         - [low] style: Unnecessary variable assignment
      Suggestions:
         - Use more descriptive variable names
         - Leverage Python built-in functions
      Positive Aspects:
         ✓ Clear function structure
         ✓ Handles basic calculation correctly

Key Takeaways
-------------

**1. Flexible Return Types**
   - Agents can return any type: strings, dictionaries, custom objects
   - Structure your returns to match your domain needs
   - Include metadata and context in responses

**2. Specialized System Prompts**
   - Tailor prompts to specific agent functions
   - Include output format specifications
   - Provide clear instructions for consistent behavior

**3. Error Handling Strategies**
   - Always include fallback behaviors for parsing failures
   - Provide meaningful error information
   - Maintain response structure even in error cases

**4. Metadata-Driven Behavior**
   - Use metadata to modify agent behavior without changing the API
   - Allow runtime configuration through metadata
   - Maintain backward compatibility

**5. JSON Response Patterns**
   - Use structured JSON for complex outputs
   - Include parsing safeguards and fallbacks
   - Enhance LLM responses with additional metadata

**6. Domain Specialization**
   - Each agent should have a clear, focused purpose
   - Customize processing logic for domain requirements
   - Provide domain-specific validation and enhancement

Common Patterns for Custom Agents
----------------------------------

**1. Validation Agent**:

.. code-block:: python

   class ValidationAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> Dict[str, bool]:
           # Validate input against business rules
           return {
               "is_valid": True/False,
               "violations": ["list of violations"],
               "confidence": 0.95
           }

**2. Classification Agent**:

.. code-block:: python

   class ClassificationAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Classify input into categories
           return {
               "primary_category": "category_name",
               "confidence": 0.87,
               "secondary_categories": ["alt1", "alt2"],
               "reasoning": "explanation"
           }

**3. Summarization Agent**:

.. code-block:: python

   class SummarizationAgent(BaseAgent):
       async def process(self, input: IAgentInput) -> Dict[str, Any]:
           # Generate structured summaries
           return {
               "summary": "concise summary",
               "key_points": ["point1", "point2"],
               "original_length": len(input.message),
               "compression_ratio": 0.25
           }

Next Steps
----------

After mastering custom agents:

1. **Example 03**: Learn memory patterns for context-aware agents
2. **Example 04**: Add tool integration for external capabilities
3. **Example 05**: Compose multiple custom agents into systems
4. **Example 06**: Test custom agents thoroughly

**Design Considerations**:

- Plan your return types based on how the data will be consumed
- Include enough metadata for debugging and monitoring
- Design error handling that maintains usability
- Consider how agents will interact in larger systems

This example shows how the simple agent pattern scales to sophisticated, domain-specific functionality while maintaining clarity and testability.