import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize the chatbot with a pre-trained conversational model
        """
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Initialize chat history
            self.chat_history_ids = None
            self.chat_history_attention_mask = None
            self.max_history_length = 4  # Keep last 2 exchanges
            
            # Set pad token if not already set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def generate_response(self, user_input):
        """
        Generate a response based on user input and previous context
        """
        try:
            # Encode the new user input with attention mask
            new_user_input = self.tokenizer.encode(
                user_input + self.tokenizer.eos_token, 
                return_tensors='pt'
            )
            new_attention_mask = torch.ones_like(new_user_input)

            # Append to chat history
            if self.chat_history_ids is not None:
                bot_input_ids = torch.cat([self.chat_history_ids, new_user_input], dim=-1)
                attention_mask = torch.cat([self.chat_history_attention_mask, new_attention_mask], dim=-1)
                
                # Truncate history if too long
                if bot_input_ids.shape[-1] > 1000:
                    keep_length = 1000 - new_user_input.shape[-1]
                    bot_input_ids = torch.cat([self.chat_history_ids[:, -keep_length:], new_user_input], dim=-1)
                    attention_mask = torch.cat([self.chat_history_attention_mask[:, -keep_length:], new_attention_mask], dim=-1)
            else:
                bot_input_ids = new_user_input
                attention_mask = new_attention_mask

            # Generate a response with improved parameters
            chat_history_ids = self.model.generate(
                bot_input_ids, 
                attention_mask=attention_mask,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,          # More focused than 100
                top_p=0.9,         # Slightly higher for better diversity
                temperature=0.7,   # Slightly lower for more coherent responses
                repetition_penalty=1.2  # Prevent repetitive responses
            )

            # Decode and clean the response
            response = self.tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
                skip_special_tokens=True
            ).strip()
            
            # Remove any trailing incomplete sentences
            if '.' in response:
                response = response[:response.rfind('.')+1]

            # Update chat history
            self.chat_history_ids = chat_history_ids
            self.chat_history_attention_mask = torch.ones_like(chat_history_ids)
            
            return response if response else "I'm not sure how to respond to that."

        except Exception as e:
            print(f"Error generating response: {e}")
            # Reset conversation history on error
            self.chat_history_ids = None
            self.chat_history_attention_mask = None
            return "I'm having trouble responding. Could you repeat that?"

def main():
    print("Xydle AI - Type 'exit' to quit.")
    
    try:
        # Initialize the chatbot
        chatbot = HuggingFaceChatbot()
        
        # Chat loop
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input:
                    continue
                
                response = chatbot.generate_response(user_input)
                print("AI:", response)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
                
    except Exception as e:
        print(f"Failed to initialize chatbot: {e}")

if __name__ == "__main__":
    main()
