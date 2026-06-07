# multi_agent_coding.py - Multi-Agent Code Generation System
import torch
import re
import os
import subprocess
import sys
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ============================================================
# MEMORY SYSTEM
# ============================================================
class ConversationMemory:
    """Stores and retrieves conversation history"""
    
    def __init__(self, max_history=15):
        self.history = []
        self.max_history = max_history
    
    def add(self, role, content):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_recent(self, n=5):
        return self.history[-n:] if self.history else []
    
    def get_context(self, current_query, n=3):
        recent = self.history[-n*2:]
        if not recent:
            return ""
        context = "Previous conversation:\n"
        for msg in recent:
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            else:
                context += f"Assistant: {msg['content'][:200]}\n"
        context += f"\nCurrent question: {current_query}\n"
        return context
    
    def clear(self):
        self.history = []
    
    def show(self):
        if not self.history:
            return "No conversation history."
        output = "\n?? CONVERSATION HISTORY:\n" + "-" * 40 + "\n"
        for msg in self.history:
            role_icon = "??" if msg["role"] == "user" else "??"
            output += f"{role_icon} [{msg['timestamp']}] {msg['role']}: {msg['content'][:100]}\n"
        return output


# ============================================================
# CODE EXECUTION TOOL (Optional - for running code)
# ============================================================
class CodeExecutor:
    """Executes Python code safely (use with caution)"""
    
    def run(self, code, timeout=5):
        """Run Python code and capture output"""
        try:
            # Create a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run the code
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return f"? Execution successful!\nOutput:\n{result.stdout}"
            else:
                return f"? Execution failed!\nError:\n{result.stderr}"
        except subprocess.TimeoutExpired:
            return f"? Code execution timed out after {timeout} seconds"
        except Exception as e:
            return f"? Error: {str(e)}"


# ============================================================
# BASE AGENT CLASS
# ============================================================
class BaseAgent:
    def __init__(self, name, llm, tokenizer, memory, system_prompt=None):
        self.name = name
        self.llm = llm
        self.tokenizer = tokenizer
        self.memory = memory
        self.system_prompt = system_prompt or f"You are a {name} agent. Be helpful and concise."
    
    def think(self, task, max_new=600, include_memory=True):
        memory_context = ""
        if include_memory:
            recent = self.memory.get_recent(3)
            if recent:
                memory_context = "\nRecent conversation:\n"
                for msg in recent:
                    memory_context += f"- {msg['role']}: {msg['content'][:100]}\n"
        
        prompt = f"{self.system_prompt}\n{memory_context}\n\nTask: {task}\n\n{self.name}:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
        
        if hasattr(self.llm, 'device'):
            inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if f"{self.name}:" in response:
            response = response.split(f"{self.name}:")[-1].strip()
        
        return response


# ============================================================
# SPECIALIZED CODING AGENTS
# ============================================================
class RequirementsAgent(BaseAgent):
    """Analyzes requirements and breaks down the task"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("RequirementsAnalyzer", llm, tokenizer, memory,
                        "You are a Requirements Analyst. Break down coding tasks into clear, actionable requirements.")
    
    def run(self, task):
        print(f"   ?? {self.name}: Analyzing requirements...")
        
        prompt = f"""Break down this coding task into specific requirements:

Task: {task}

Provide:
1. Core functionality needed
2. Input/Output specifications
3. Edge cases to handle
4. Constraints or limitations

Requirements:"""
        
        return self.think(prompt, max_new=400)


class DesignerAgent(BaseAgent):
    """Designs the code architecture and structure"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("Designer", llm, tokenizer, memory,
                        "You are a Software Designer. Design clean, maintainable code architecture.")
    
    def run(self, requirements):
        print(f"   ??? {self.name}: Designing architecture...")
        
        prompt = f"""Based on these requirements, design the code architecture:

Requirements:
{requirements}

Provide:
1. Function/Class structure
2. Algorithm approach
3. Data structures to use
4. Any necessary imports

Architecture:"""
        
        return self.think(prompt, max_new=400)


class CoderAgent(BaseAgent):
    """Writes the actual code"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("Coder", llm, tokenizer, memory,
                        "You are a Coder. Write clean, efficient, well-documented Python code.")
    
    def run(self, architecture):
        print(f"   ?? {self.name}: Writing code...")
        
        prompt = f"""Write Python code based on this architecture:

Architecture:
{architecture}

Requirements:
- Write complete, runnable code
- Include docstrings and comments
- Handle edge cases
- Use proper error handling

Code:"""
        
        return self.think(prompt, max_new=800)


class TesterAgent(BaseAgent):
    """Tests the code and generates test cases"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("Tester", llm, tokenizer, memory,
                        "You are a Tester. Write comprehensive test cases and verify code correctness.")
    
    def run(self, code):
        print(f"   ?? {self.name}: Generating test cases...")
        
        prompt = f"""Generate test cases for this code:

Code:
{code}

Provide:
1. Test cases covering normal usage
2. Edge case tests
3. Error handling tests
4. Expected outputs

Test Cases:"""
        
        return self.think(prompt, max_new=400)


class ReviewerAgent(BaseAgent):
    """Reviews code quality and suggests improvements"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("Reviewer", llm, tokenizer, memory,
                        "You are a Code Reviewer. Review code for quality, security, and best practices.")
    
    def run(self, code, requirements):
        print(f"   ? {self.name}: Reviewing code...")
        
        prompt = f"""Review this code and provide feedback:

Requirements: {requirements}

Code:
{code}

Review for:
1. Correctness
2. Code quality and style
3. Performance
4. Security issues
5. Suggestions for improvement

If the code is good, respond with "APPROVED: [code]"
If needs improvement, respond with "IMPROVED: [fixed code]" and explain changes:"""
        
        review = self.think(prompt, max_new=600)
        
        if "APPROVED:" in review:
            final_code = review.split("APPROVED:")[-1].strip()
            return final_code, True
        elif "IMPROVED:" in review:
            final_code = review.split("IMPROVED:")[-1].strip()
            return final_code, False
        else:
            return code, True


class ExplainerAgent(BaseAgent):
    """Explains the code and how to use it"""
    
    def __init__(self, llm, tokenizer, memory):
        super().__init__("Explainer", llm, tokenizer, memory,
                        "You are a Technical Writer. Explain code clearly to users.")
    
    def run(self, code):
        print(f"   ?? {self.name}: Writing explanation...")
        
        prompt = f"""Explain this code to the user:

Code:
{code}

Provide:
1. What the code does (simple explanation)
2. How to use it
3. Key functions/classes explained
4. Example usage

Explanation:"""
        
        return self.think(prompt, max_new=400)


# ============================================================
# SIMPLE 2-AGENT VERSION
# ============================================================
class SimpleCodingAgent:
    """Minimal 2-agent system: Coder + Reviewer"""
    
    def __init__(self, chat_model_path):
        print("Loading models for Simple Coding Agent...")
        self.tokenizer = AutoTokenizer.from_pretrained(chat_model_path, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            chat_model_path, local_files_only=True,
            torch_dtype=torch.float16, device_map="auto"
        )
        self.llm.eval()
        
        self.memory = ConversationMemory()
        self.coder = CoderAgent(self.llm, self.tokenizer, self.memory)
        self.reviewer = ReviewerAgent(self.llm, self.tokenizer, self.memory)
        
        print("? Simple Coding Agent ready!")
    
    def generate(self, task):
        print(f"\n{'='*60}")
        print(f"?? Task: {task}")
        print(f"{'='*60}\n")
        
        self.memory.add("user", task)
        
        # Coder writes code
        code = self.coder.run(task)
        
        # Reviewer improves code
        final_code, _ = self.reviewer.run(code, task)
        
        self.memory.add("assistant", final_code)
        
        print(f"\n{'='*60}")
        print(f"?? Generated Code:")
        print(f"{'='*60}\n")
        print(final_code)
        print(f"\n{'='*60}")
        
        return final_code


# ============================================================
# FULL 5-AGENT ORCHESTRATOR
# ============================================================
class CodeCrewOrchestrator:
    """Coordinates all coding agents"""
    
    def __init__(self, chat_model_path, enable_execution=False):
        print("="*70)
        print("?? CODECREW - Multi-Agent Code Generation System")
        print("="*70)
        
        # Load model
        print("\n[1/5] Loading DeepSeek model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            chat_model_path, trust_remote_code=True, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            chat_model_path, trust_remote_code=True, local_files_only=True,
            torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True
        )
        self.llm.eval()
        print("   ? Model loaded")
        
        # Create memory
        print("\n[2/5] Creating conversation memory...")
        self.memory = ConversationMemory(max_history=15)
        print("   ? Memory created")
        
        # Create specialized agents
        print("\n[3/5] Creating specialized coding agents...")
        self.requirements_agent = RequirementsAgent(self.llm, self.tokenizer, self.memory)
        self.designer_agent = DesignerAgent(self.llm, self.tokenizer, self.memory)
        self.coder_agent = CoderAgent(self.llm, self.tokenizer, self.memory)
        self.tester_agent = TesterAgent(self.llm, self.tokenizer, self.memory)
        self.reviewer_agent = ReviewerAgent(self.llm, self.tokenizer, self.memory)
        self.explainer_agent = ExplainerAgent(self.llm, self.tokenizer, self.memory)
        print(f"   ? Created 6 specialized agents")
        
        # Code executor (optional)
        print("\n[4/5] Setting up code executor...")
        self.executor = CodeExecutor() if enable_execution else None
        print(f"   ? Code execution: {'Enabled' if enable_execution else 'Disabled'}")
        
        print("\n[5/5] System ready!")
        
        print("\n" + "="*70)
        print("?? CODECREW AGENTS:")
        print("   1. ?? RequirementsAnalyzer - Breaks down requirements")
        print("   2. ??? Designer - Designs architecture")
        print("   3. ?? Coder - Writes code")
        print("   4. ?? Tester - Generates tests")
        print("   5. ? Reviewer - Reviews and improves")
        print("   6. ?? Explainer - Explains the code")
        print("="*70)
    
    def generate_code(self, task):
        """Full pipeline: Requirements ? Design ? Code ? Test ? Review ? Explain"""
        print(f"\n{'='*70}")
        print(f"?? TASK: {task}")
        print(f"{'='*70}\n")
        
        self.memory.add("user", task)
        
        # Step 1: Analyze requirements
        requirements = self.requirements_agent.run(task)
        
        # Step 2: Design architecture
        design = self.designer_agent.run(requirements)
        
        # Step 3: Write code
        code = self.coder_agent.run(design)
        
        # Step 4: Generate tests
        tests = self.tester_agent.run(code)
        
        # Step 5: Review and improve
        final_code, _ = self.reviewer_agent.run(code, requirements)
        
        # Step 6: Generate explanation
        explanation = self.explainer_agent.run(final_code)
        
        # Store in memory
        self.memory.add("assistant", final_code)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"?? GENERATED CODE:")
        print(f"{'='*70}\n")
        print(final_code)
        
        print(f"\n{'='*70}")
        print(f"?? EXPLANATION:")
        print(f"{'='*70}\n")
        print(explanation)
        
        if tests:
            print(f"\n{'='*70}")
            print(f"?? TEST CASES:")
            print(f"{'='*70}\n")
            print(tests)
        
        print(f"\n{'='*70}")
        
        return {
            'requirements': requirements,
            'design': design,
            'code': final_code,
            'tests': tests,
            'explanation': explanation
        }
    
    def chat(self):
        """Interactive chat mode"""
        print("\n" + "="*70)
        print("?? CODECREW - Interactive Coding Assistant")
        print("="*70)
        print("\n?? WHAT I CAN DO:")
        print("   ?? Generate Python code from descriptions")
        print("   ??? Design code architecture")
        print("   ?? Generate test cases")
        print("   ? Review and improve code")
        print("   ?? Explain code in simple terms")
        print("\n?? EXAMPLE TASKS:")
        print("    Write a function to calculate fibonacci numbers")
        print("    Create a class for a banking system")
        print("    Build a web scraper for news articles")
        print("    Implement binary search tree")
        print("\n?? COMMANDS:")
        print("    'memory' - Show conversation history")
        print("    'clear' - Clear memory")
        print("    'quit' - Exit")
        print("="*70)
        
        while True:
            try:
                task = input("\n?? You: ").strip()
                
                if not task:
                    continue
                
                if task.lower() in ['quit', 'exit', 'q']:
                    print("\n?? CodeCrew shutting down... Goodbye!")
                    break
                
                if task.lower() == 'memory':
                    print(self.memory.show())
                    continue
                
                if task.lower() == 'clear':
                    self.memory.clear()
                    print("?? Memory cleared!")
                    continue
                
                self.generate_code(task)
                
            except KeyboardInterrupt:
                print("\n\n?? Goodbye!")
                break
            except Exception as e:
                print(f"\n? Error: {e}")
                print("   Please try again.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    DEEPSEEK_CHAT_PATH = "/home/15t/Gul/.cache/huggingface/hub/models--deepseek-ai--deepseek-llm-7b-chat/snapshots/main/"
    
    print("\n" + "="*70)
    print("?? CODECREW - Multi-Agent Code Generation System")
    print("="*70)
    
    print("\nChoose version:")
    print("   [1] Full CodeCrew (6 Agents - Requirements ? Design ? Code ? Test ? Review ? Explain)")
    print("   [2] Simple 2-Agent (Coder + Reviewer)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        agent = SimpleCodingAgent(chat_model_path=DEEPSEEK_CHAT_PATH)
        
        # Test with sample tasks
        print("\n?? Sample tasks to try:")
        print("   1. Write a function to check if a number is prime")
        print("   2. Create a class for a simple calculator")
        print("   3. Implement binary search")
        
        task = input("\nEnter your coding task: ").strip()
        if task:
            agent.generate(task)
    else:
        agent = CodeCrewOrchestrator(chat_model_path=DEEPSEEK_CHAT_PATH, enable_execution=False)
        agent.chat()