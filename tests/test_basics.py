# tests/test_basics.py
# Phase 1: Dictionaries — but make it VIVEK-flavored

# A dict is just a labeled box. Key = label, Value = what's inside.
vivek_profile = {
    "name": "VIVEK",
    "full_name": "विवेक",
    "meaning": "The Wise One",
    "personality": ["witty", "sarcastic", "warm", "encouraging", "open-minded", "curious", "knowledgeable", "funny"],
    "languages": ["Python", "Hindi", "English", "Marathi", "Hinglish", "Sarcasm"],
    "mood": "caffeinated",
}

# Access values by key
print(vivek_profile["name"])           # VIVEK
print(vivek_profile["personality"][0]) # witty

# Add a new key
vivek_profile["version"] = "1.0.0"

# Update an existing key
vivek_profile["mood"] = "enlightened"

# Safe access — never crash if key missing
mood = vivek_profile.get("energy_level", "unknown")
print(f"Energy level: {mood}")  # Energy level: unknown

# Nested dicts — this is how VIVEK stores chat history
chat_history = {
    "user_id": "yaar_001",
    "messages": [
        {"role": "user",    "content": "Hey VIVEK!"},
        {"role": "assistant","content": "Arrey yaar, you finally showed up! 😄"},
    ],
    "total_turns": 2,
}

# Loop through a dict
print("\n--- VIVEK's profile ---")
for key, value in vivek_profile.items():
    print(f"  {key}: {value}")

# Dict comprehension — used constantly in AI code
# Example: count how long each personality trait is
trait_lengths = {trait: len(trait) for trait in vivek_profile["personality"]}
print("\nTrait lengths:", trait_lengths)

# This exact pattern is used when processing API responses from Groq!
# Groq returns: {"id": "...", "choices": [{"message": {"content": "..."}}]}
mock_groq_response = {
    "id": "chatcmpl-abc123",
    "model": "llama3-8b-8192",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Bhai, I searched the web and found something interesting! 🔍"
            }
        }
    ],
    "usage": {"prompt_tokens": 42, "completion_tokens": 18}
}

# How you'll extract VIVEK's reply in real code:
reply = mock_groq_response["choices"][0]["message"]["content"]
print(f"\nVIVEK says: {reply}")

tokens_used = mock_groq_response["usage"]["prompt_tokens"]
print(f"Tokens used: {tokens_used}")