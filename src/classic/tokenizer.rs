//! Zero-dependency deterministic Tokenizer for micro-language model (V = 21).
//! Supports arithmetic tokens (0..5, +, -, =), Russian words (кот, пес, животное, друг, это, да, нет, кто),
//! and control tokens (<pad>, <eos>, <user>, <bot>).

/// Static vocabulary mapping of exactly 21 tokens (v1).
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

/// Extended vocabulary mapping of 41 tokens (v2).
/// Tokens 0..20 are 100% backward compatible with v1.
pub const VOCAB_V2: [&str; 41] = [
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
    "6",         // 21
    "7",         // 22
    "8",         // 23
    "9",         // 24
    "*",         // 25
    "волк",      // 26
    "лиса",      // 27
    "заяц",      // 28
    "рыба",      // 29
    "птица",     // 30
    "зверь",     // 31
    "хищник",    // 32
    "человек",   // 33
    "враг",      // 34
    "где",       // 35
    "что",       // 36
    "река",      // 37
    "небо",      // 38
    "лес",       // 39
    "дом",       // 40
];

/// Extended vocabulary mapping of 53 tokens (v3).
/// Tokens 0..40 are 100% backward compatible with v2 and v1.
pub const VOCAB_V3: [&str; 53] = [
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
    "6",         // 21
    "7",         // 22
    "8",         // 23
    "9",         // 24
    "*",         // 25
    "волк",      // 26
    "лиса",      // 27
    "заяц",      // 28
    "рыба",      // 29
    "птица",     // 30
    "зверь",     // 31
    "хищник",    // 32
    "человек",   // 33
    "враг",      // 34
    "где",       // 35
    "что",       // 36
    "река",      // 37
    "небо",      // 38
    "лес",       // 39
    "дом",       // 40
    "/",         // 41
    "медведь",   // 42
    "змея",      // 43
    "щука",      // 44
    "дуб",       // 45
    "дерево",    // 46
    "тайга",     // 47
    "нора",      // 48
    "поле",      // 49
    "трава",     // 50
    "вода",      // 51
    "ест",       // 52
];

/// Chinchilla vocabulary mapping of exactly 65 tokens.
/// Tokens 0..52 are 100% backward compatible with v3, v2, and v1.
/// Adds logical connectives (не, как, почему, если, то, или) and foundational
/// science, mathematics, AI, and reasoning concepts (наука, число, модель, разум, знание, логика).
pub const VOCAB_CHINCHILLA: [&str; 65] = [
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
    "6",         // 21
    "7",         // 22
    "8",         // 23
    "9",         // 24
    "*",         // 25
    "волк",      // 26
    "лиса",      // 27
    "заяц",      // 28
    "рыба",      // 29
    "птица",     // 30
    "зверь",     // 31
    "хищник",    // 32
    "человек",   // 33
    "враг",      // 34
    "где",       // 35
    "что",       // 36
    "река",      // 37
    "небо",      // 38
    "лес",       // 39
    "дом",       // 40
    "/",         // 41
    "медведь",   // 42
    "змея",      // 43
    "щука",      // 44
    "дуб",       // 45
    "дерево",    // 46
    "тайга",     // 47
    "нора",      // 48
    "поле",      // 49
    "трава",     // 50
    "вода",      // 51
    "ест",       // 52
    "не",        // 53
    "как",       // 54
    "почему",    // 55
    "если",      // 56
    "то",        // 57
    "или",       // 58
    "наука",     // 59
    "число",     // 60
    "модель",    // 61
    "разум",     // 62
    "знание",    // 63
    "логика",    // 64
];

/// Alias for VOCAB_CHINCHILLA (v4 vocabulary).
pub const VOCAB_V4: [&str; 65] = VOCAB_CHINCHILLA;

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
    /// Creates a new Tokenizer pre-populated with the extended 53-token vocabulary (v3).
    pub fn new() -> Self {
        Self::v3()
    }

    /// Creates a Tokenizer pre-populated with the Chinchilla 65-token vocabulary (v4).
    pub fn chinchilla() -> Self {
        let vocab = VOCAB_CHINCHILLA.iter().map(|&s| s.to_string()).collect();
        Self { vocab }
    }

    /// Creates a Tokenizer pre-populated with the legacy 21-token vocabulary (v1).
    pub fn v1() -> Self {
        let vocab = VOCAB.iter().map(|&s| s.to_string()).collect();
        Self { vocab }
    }

    /// Creates a Tokenizer pre-populated with the extended 41-token vocabulary (v2).
    pub fn v2() -> Self {
        let vocab = VOCAB_V2.iter().map(|&s| s.to_string()).collect();
        Self { vocab }
    }

    /// Creates a Tokenizer pre-populated with the extended 53-token vocabulary (v3).
    pub fn v3() -> Self {
        let vocab = VOCAB_V3.iter().map(|&s| s.to_string()).collect();
        Self { vocab }
    }

    /// Creates a custom Tokenizer from an arbitrary vocabulary slice.
    pub fn from_vocab(vocab_slice: &[&str]) -> Self {
        let vocab = vocab_slice.iter().map(|&s| s.to_string()).collect();
        Self { vocab }
    }

    /// Returns the total vocabulary size.
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

            // Check for single character tokens (0..9, +, -, =, *, /)
            let single_char_str = &text[i..i + c.len_utf8()];
            if let Some(id) = self.token_to_id(single_char_str) {
                tokens.push(id);
                chars.next();
                continue;
            }

            // Word tokens (Russian words)
            let mut end_idx = i + c.len_utf8();
            chars.next();
            while let Some(&(next_i, next_c)) = chars.peek() {
                if next_c.is_whitespace()
                    || next_c == '<'
                    || next_c == '>'
                    || next_c == '?'
                    || next_c == '.'
                    || next_c == ','
                    || next_c == '!'
                    || next_c == ':'
                    || next_c == ';'
                    || next_c == '"'
                    || next_c == '\''
                    || next_c == '('
                    || next_c == ')'
                    || next_c == '['
                    || next_c == ']'
                    || matches!(next_c, '0'..='9' | '+' | '-' | '=' | '*' | '/')
                {
                    break;
                }
                end_idx = next_i + next_c.len_utf8();
                chars.next();
            }

            let word = &text[i..end_idx];
            if let Some(id) = self.token_to_id(word) {
                tokens.push(id);
            } else {
                let norm = word.to_lowercase().replace('ё', "е");
                if let Some(id) = self.token_to_id(&norm) {
                    tokens.push(id);
                }
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
        assert_eq!(tok.vocab_size(), 53);
        let tok_chin = Tokenizer::chinchilla();
        assert_eq!(tok_chin.vocab_size(), 65);
        let tok_v2 = Tokenizer::v2();
        assert_eq!(tok_v2.vocab_size(), 41);
        let tok_v1 = Tokenizer::v1();
        assert_eq!(tok_v1.vocab_size(), 21);
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

    #[test]
    fn test_roundtrip_v2_arithmetic_multiplication() {
        let tok = Tokenizer::new();
        let original = "<user> 2 * 3 = <bot> 6 <eos>";
        let encoded = tok.encode(original);
        assert_eq!(encoded, vec![2, 6, 25, 7, 12, 3, 21, 1]);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_roundtrip_v2_spatial_and_taxonomy() {
        let tok = Tokenizer::new();
        let original = "<user> где волк <bot> лес <eos>";
        let encoded = tok.encode(original);
        assert_eq!(encoded, vec![2, 35, 26, 3, 39, 1]);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, original);

        let fact = "лиса это хищник <eos>";
        let enc_fact = tok.encode(fact);
        assert_eq!(enc_fact, vec![27, 17, 32, 1]);
        assert_eq!(tok.decode(&enc_fact), fact);
    }

    #[test]
    fn test_roundtrip_v3_division_and_predicates() {
        let tok = Tokenizer::new();
        let original = "<user> 6 / 2 = <bot> 3 <eos>";
        let encoded = tok.encode(original);
        assert_eq!(encoded, vec![2, 21, 41, 6, 12, 3, 7, 1]);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, original);

        let fact = "волк ест заяц <eos>";
        let enc_fact = tok.encode(fact);
        assert_eq!(enc_fact, vec![26, 52, 28, 1]);
        assert_eq!(tok.decode(&enc_fact), fact);

        let where_pike = "<user> где щука <bot> вода <eos>";
        let enc_pike = tok.encode(where_pike);
        assert_eq!(enc_pike, vec![2, 35, 44, 3, 51, 1]);
        assert_eq!(tok.decode(&enc_pike), where_pike);
    }

    #[test]
    fn test_roundtrip_chinchilla_reasoning() {
        let tok = Tokenizer::chinchilla();
        assert_eq!(tok.vocab_size(), 65);

        // Check new tokens mapping
        assert_eq!(tok.token_to_id("не"), Some(53));
        assert_eq!(tok.token_to_id("как"), Some(54));
        assert_eq!(tok.token_to_id("почему"), Some(55));
        assert_eq!(tok.token_to_id("если"), Some(56));
        assert_eq!(tok.token_to_id("то"), Some(57));
        assert_eq!(tok.token_to_id("или"), Some(58));
        assert_eq!(tok.token_to_id("наука"), Some(59));
        assert_eq!(tok.token_to_id("число"), Some(60));
        assert_eq!(tok.token_to_id("модель"), Some(61));
        assert_eq!(tok.token_to_id("разум"), Some(62));
        assert_eq!(tok.token_to_id("знание"), Some(63));
        assert_eq!(tok.token_to_id("логика"), Some(64));

        // Science & logic sentences
        let sent = "логика это наука <eos>";
        let enc = tok.encode(sent);
        assert_eq!(enc, vec![64, 17, 59, 1]);
        assert_eq!(tok.decode(&enc), sent);

        let sent2 = "наука это знание <eos>";
        let enc2 = tok.encode(sent2);
        assert_eq!(enc2, vec![59, 17, 63, 1]);
        assert_eq!(tok.decode(&enc2), sent2);

        // If-then reasoning
        let if_then = "если волк ест заяц то волк хищник <eos>";
        let enc_if = tok.encode(if_then);
        assert_eq!(tok.decode(&enc_if), if_then);

        // Dialogue QA
        let dial = "<user> почему человек не зверь <bot> человек это разум <eos>";
        let enc_dial = tok.encode(dial);
        assert_eq!(tok.decode(&enc_dial), dial);

        // Normalization (case insensitivity and ё)
        let unnorm = "Логика это Наука <eos>";
        let enc_unnorm = tok.encode(unnorm);
        assert_eq!(enc_unnorm, vec![64, 17, 59, 1]);
    }
}
