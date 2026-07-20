/**
 * Profanity / NSFW blocklist - words the assistant never proactively SUGGESTS or
 * offers as an autocorrection, and never corrects a benign word INTO.
 *
 * GENERATED at build time by classifying the model vocab with an LLM (see
 * build_model/profanity_blocklist.json). Curated to avoid false positives (the
 * Scunthorpe problem): clinical anatomy, drug names, religious names, ordinary
 * negative words, and benign homographs are deliberately absent. One flat set -
 * profanity, slurs, and explicit sexual terms are treated identically.
 *
 * The user can re-enable any word by adding it to their personal dictionary, or
 * disable the whole filter. IMPORTANT: this filters OUR output only - a word the
 * user types themselves is never removed or corrected away (see EngineCore).
 */
export const PROFANITY: ReadonlySet<string> = new Set([
  "aryans", "asshole", "bastard", "bawdy", "bestiality", "bint", "bitches", "blackamoor",
  "bondage", "boobs", "brainfuck", "brothel", "bugger", "bullshit", "chav", "clitoral",
  "cock", "coon", "coons", "cum", "cunnilingus", "cunt", "dike", "dominatrix",
  "dyke", "ejaculation", "erotic", "fag", "faggot", "fap", "fica", "flasher",
  "fondling", "fuck", "fucked", "fucking", "gonad", "gypsies", "gypsy", "hajji",
  "honky", "hooker", "horny", "incest", "jap", "kafir", "kike", "klan",
  "kraut", "lewdness", "lust", "masturbating", "masturbation", "mestizo", "milf", "nazi",
  "negro", "negroes", "negros", "nig", "nigga", "nigger", "niggers", "niggr",
  "nigra", "nigro", "nob", "nonce", "nude", "nudes", "orgasm", "orgasms",
  "orgy", "paki", "pedophile", "pedophilia", "perversion", "pimp", "pimps", "piss",
  "pissed", "pogrom", "polak", "poof", "poon", "porn", "porno", "pornographic",
  "pornography", "prick", "prostitute", "prostitution", "pussy", "putz", "queer", "queers",
  "rape", "raped", "rapes", "raping", "rapist", "retard", "retarded", "retards",
  "screwing", "sexting", "sexy", "shit", "shite", "shitty", "shota", "skinhead",
  "slag", "slut", "smut", "smuts", "sodomy", "softcore", "squaw", "threesome",
  "tit", "tits", "waffen", "whore", "whores", "wop", "wtf",
]);

/** True if the word is blocklisted and not re-enabled by the user allowlist. */
export function isProfane(word: string, allow?: ReadonlySet<string>): boolean {
  const w = word.toLowerCase();
  if (allow && allow.has(w)) return false;
  return PROFANITY.has(w);
}
