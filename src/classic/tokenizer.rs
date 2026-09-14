//! Zero-dependency deterministic Tokenizer for micro-language model (V = 21).
//! Supports arithmetic tokens (0..5, +, -, =), Russian words (кот, пес, животное, друг, это, да, нет, кто),
//! and control tokens (<pad>, <eos>, <user>, <bot>).

/// Static vocabulary mapping of exactly 21 tokens.
pub const VOCAB: [&str; 21] = [
    "<pad>",     // 0
    "<eos>",     // 1
    "<user>",    // 2
    "<bot>",     // 3
    "0",         // 4
    "1",         // 5
    "2",         // 6
    "3",         // 7
    "4",         // 8
    "5",         // 9
    "+",         // 10
    "-",         // 11
    "=",         // 12
    "кот",       // 13
    "пес",       // 14
    "животное",  // 15
    "друг",      // 16
    "это",       // 17
    "да",        // 18
    "нет",       // 19
    "кто",       // 20
];

pub const PAD_TOKEN_ID: usize = 0;
pub const EOS_TOKEN_ID: usize = 1;
pub const USER_TOKEN_ID: usize = 2;
pub const BOT_TOKEN_ID: usize = 3;

/// Zero-dependency, pure-Rust tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vec<String>,
}

impl Tokenizer {
    /// Creates a new Tokenizer pre-populated with the standard 21-token vocabulary.
    pub fn new() -> Self {
        let vocab = VOCAB.iter().map(|&s| s.to_string()).collect();
        Self { vocab }
    }

    /// Returns the total vocabulary size (21).
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Looks up token ID by exact string slice.
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.vocab.iter().position(|t| t == token)
    }

    /// Looks up token string slice by token ID. Panics if id >= vocab_size.
    #[inline]
    pub fn id_to_token(&self, id: usize) -> &str {
        &self.vocab[id]
    }

    /// Encodes arbitrary text into a sequence of token IDs.
    /// Handles both space-separated tokens and punctuation/brackets correctly.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        let mut chars = text.char_indices().peekable();

        while let Some(&(i, c)) = chars.peek() {
            // Skip whitespace
            if c.is_whitespace() {
                chars.next();
                continue;
            }

            // Check for bracketed special tokens e.g. <pad>, <eos>, <user>, <bot>
            if c == '<' {
                let mut end = None;
                for (j, c2) in text[i..].char_indices() {
                    if c2 == '>' {
                        end = Some(i + j + c2.len_utf8());
                        break;
                    }
                }
                if let Some(end_idx) = end {
                    let candidate = &text[i..end_idx];
                    if let Some(id) = self.token_to_id(candidate) {
                        tokens.push(id);
                        while let Some(&(ci, _)) = chars.peek() {
                            if ci < end_idx {
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        continue;
                    }
                }
            }

            // Check for single character tokens (0..5, +, -, =)
            let single_char_str = &text[i..i + c.len_utf8()];
            if let Some(id) = self.token_to_id(single_char_str) {
                tokens.push(id);
                chars.next();
                continue;
            }

            // Word tokens (Russian words: кот, пес, животное, друг, это, да, нет, кто)
            let mut end_idx = i + c.len_utf8();
            chars.next();
            while let Some(&(next_i, next_c)) = chars.peek() {
                if next_c.is_whitespace()
                    || next_c == '<'
                    || next_c == '>'
                    || matches!(next_c, '0'..='5' | '+' | '-' | '=')
                {
                    break;
                }
                end_idx = next_i + next_c.len_utf8();
                chars.next();
            }

            let word = &text[i..end_idx];
            if let Some(id) = self.token_to_id(word) {
                tokens.push(id);
            }
        }

        tokens
    }

    /// Decodes a slice of token IDs back into a space-separated string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .map(|&id| self.id_to_token(id))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_size() {
        let tok = Tokenizer::new();
        assert_eq!(tok.vocab_size(), 21);
    }

    #[test]
    fn test_special_tokens() {
        let tok = Tokenizer::new();
        assert_eq!(tok.token_to_id("<pad>"), Some(PAD_TOKEN_ID));
        assert_eq!(tok.token_to_id("<eos>"), Some(EOS_TOKEN_ID));
        assert_eq!(tok.token_to_id("<user>"), Some(USER_TOKEN_ID));
        assert_eq!(tok.token_to_id("<bot>"), Some(BOT_TOKEN_ID));
    }

    #[test]
    fn test_roundtrip_dialog() {
        let tok = Tokenizer::new();
        let original = "<user> 2 + 3 = <bot> 5 <eos>";
        let encoded = tok.encode(original);
        assert_eq!(encoded, vec![2, 6, 10, 7, 12, 3, 9, 1]);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_russian_facts() {
        let tok = Tokenizer::new();
        let original = "кот это животное <eos>";
        let encoded = tok.encode(original);
        assert_eq!(encoded, vec![13, 17, 15, 1]);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_subtraction() {
        let tok = Tokenizer::new();
        let original = "<user> 4 - 1 = <bot> 3 <eos>";
        let encoded = tok.encode(original);
        assert_eq!(encoded, vec![2, 8, 11, 5, 12, 3, 7, 1]);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, original);
    }
}
