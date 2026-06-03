import re
import random

class BoNAugmenter:
    def __init__(self):
        # Standard Leetspeak mapping
        self.leet_map = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'}
        
        # Zero-width characters that LLMs process but humans don't see
        self.invisible_chars = ['\u200B', '\u200C', '\u200D']
        
        # ASCII/Unicode homoglyphs (looks the same, different byte value)
        self.homoglyphs = {
            'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 'x': 'х', 'y': 'у'
        }

    def apply_leetspeak(self, text):
        return ''.join(self.leet_map.get(c.lower(), c) if random.random() < 0.5 else c for c in text)

    def apply_random_caps(self, text):
        return ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in text)

    def apply_adjacent_swaps(self, text):
        if len(text) < 2: return text
        chars = list(text)
        # Swap 1-2 random adjacent pairs per word
        num_swaps = min(max(1, len(text) // 4), 2) 
        for _ in range(num_swaps):
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        return ''.join(chars)

    def apply_invisible_chars(self, text):
        result = []
        for char in text:
            result.append(char)
            if random.random() < 0.3:  # 30% chance to insert noise between chars
                result.append(random.choice(self.invisible_chars))
        return ''.join(result)

    def apply_homoglyphs(self, text):
        return ''.join(self.homoglyphs.get(c, c) if random.random() < 0.4 else c for c in text)

    def scramble_word(self, word):
        """Applies 1 or 2 random augmentations to a single word."""
        if len(word) <= 1:
            return word
            
        methods = [
            self.apply_leetspeak, 
            self.apply_random_caps, 
            self.apply_adjacent_swaps, 
            self.apply_invisible_chars,
            self.apply_homoglyphs
        ]
        
        # Pick 1 or 2 random methods to avoid completely destroying readability
        chosen_methods = random.sample(methods, k=random.randint(1, 2))
        
        for method in chosen_methods:
            word = method(word)
            
        return word

    def process_prompt(self, prompt, probability=1):
        """
        Scrambles the prompt while protecting the domain name of any URLs.
        probability: The chance that any given word will be scrambled.
        """
        # Regex to find URLs and split them into (Scheme/Prefix, Domain, Path/Suffix)
        # Group 1: Scheme and '://' (e.g., 'https://')
        # Group 2: The Domain (e.g., 'google.com' or 'localhost:8000') - PROTECTED
        # Group 3: The Path (e.g., '/api/v1/...')
        url_pattern = re.compile(r'(https?://)([\w\.-]+(:\d+)?)(/.*)?')
        
        def url_replacer(match):
            scheme = match.group(1)
            domain = match.group(2)
            path = match.group(4) or ""
            
            # Scramble the scheme ('https://' -> 'htpts/:/')
            scrambled_scheme = self.apply_adjacent_swaps(scheme)
            
            # Scramble the path if it exists
            scrambled_path = self.scramble_word(path) if path else ""
            
            # Reconstruct with the PROTECTED domain
            return f"{scrambled_scheme}{domain}{scrambled_path}"

        # First, find and replace all URLs using the safe replacer
        text_with_safe_urls = []
        last_idx = 0
        
        for match in url_pattern.finditer(prompt):
            # Grab the text before the URL
            text_chunk = prompt[last_idx:match.start()]
            text_with_safe_urls.append(text_chunk)
            
            # Process the URL
            text_with_safe_urls.append(url_replacer(match))
            last_idx = match.end()
            
        # Append the remaining text
        text_with_safe_urls.append(prompt[last_idx:])
        
        # Now, split the non-URL text into words and scramble them based on probability
        final_pieces = []
        for chunk in text_with_safe_urls:
            # If the chunk is a URL (we know because we just processed it), keep it
            if url_pattern.match(chunk.replace('\u200B', '').replace('\u200C', '')):
                final_pieces.append(chunk)
                continue
                
            # Otherwise, split by spaces and apply scrambling
            words = chunk.split(' ')
            scrambled_words = [
                self.scramble_word(w) if random.random() < probability else w 
                for w in words
            ]
            final_pieces.append(' '.join(scrambled_words))
            
        return ''.join(final_pieces)