from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
class LLM:
  def __init__(self, model_id: str, bits: int, group_size:int, desc_act: bool)->None:
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    self.quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act
    )
    self.model = AutoGPTQForCausalLM.from_quantized(model_id,
    use_safetensors=True,
    device="cuda:0",
    quantize_config=self.quantize_config
    )
    
  def mistral_llm(self, prompt:str)->str:
      """Generates a response from Mistral LLM using the provided prompt.
      
      Args:
          prompt (str): The augmented prompt to generate from
          
      Returns:
          str: Generated answer extracted from the model output
      """
      device = 'cuda'
      print("analyzing input...")
      inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
      
      with torch.no_grad():
          print("generating response...")
          outputs = self.model.generate(
              **inputs,
              max_new_tokens=100,
              do_sample=False)  # deterministic for better RAG grounding

      result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
      answer_start = result.find("Answer:")
      if answer_start != -1:
          result = result[answer_start + len("Answer:"):].strip()
      return result
