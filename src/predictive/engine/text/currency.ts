/**
 * Currency tidying: turn "$1000" into "$1,000", "1000 dollars" into "$1,000", and place the symbol
 * where that currency conventionally sits. Pure and unit-tested; the plugin runs it on a boundary
 * and offers it in the suggestion popup.
 *
 * Two independent behaviours, each gated by the caller:
 *   - format:       reflow an amount that already has a symbol (group the thousands, move the symbol
 *                   to the side that currency normally uses).
 *   - wordToSymbol: replace a spelled-out currency word/code after a number with its symbol.
 *
 * The user picks only the THOUSANDS separator (comma / period / none). The decimal mark follows from
 * it (the other of . and ,), except with "none", where whatever the user typed is preserved. The
 * symbol SIDE is a property of the currency, not a setting: "$100" but "100 kr".
 */

export interface CurrencyStyle {
  /** Groups-of-three separator: "," , "." or "" (none). */
  thousands: string;
  /** Decimal mark to emit: "." , "," , or null = keep whatever the user typed. */
  decimal: string | null;
}

/** Map the one user setting (thousands separator) to a full style. */
export function currencyStyleFor(thousands: "comma" | "period" | "none"): CurrencyStyle {
  if (thousands === "period") return { thousands: ".", decimal: "," }; // 1.000,50
  if (thousands === "none") return { thousands: "", decimal: null }; //   1000.50 or 1000,50 (kept)
  return { thousands: ",", decimal: "." }; //                              1,000.50
}

/**
 * Spelled-out currency words / ISO codes to symbol. Lower-cased keys, matched case-insensitively.
 * Several currencies legitimately share a sign ($ for USD/CAD/AUD/…, ¥ for JPY/CNY) - the sign is
 * what the writer wants on the page.
 */
const WORD_TO_SYMBOL: Record<string, string> = {
  dollar: "$", dollars: "$", usd: "$", buck: "$", bucks: "$",
  cad: "$", aud: "$", nzd: "$", sgd: "$", hkd: "$", mxn: "$", peso: "$", pesos: "$",
  euro: "€", euros: "€", eur: "€",
  pound: "£", pounds: "£", gbp: "£", quid: "£", sterling: "£",
  yen: "¥", jpy: "¥", yuan: "¥", cny: "¥", rmb: "¥", renminbi: "¥",
  rupee: "₹", rupees: "₹", inr: "₹",
  franc: "Fr", francs: "Fr", chf: "Fr",
  ruble: "₽", rubles: "₽", rouble: "₽", roubles: "₽", rub: "₽",
  won: "₩", krw: "₩",
  lira: "₺", try: "₺",
  real: "R$", reais: "R$", brl: "R$",
  rand: "R", zar: "R",
  krona: "kr", kronor: "kr", krone: "kr", kroner: "kr", sek: "kr", nok: "kr", dkk: "kr",
  zloty: "zł", zlotych: "zł", pln: "zł",
  baht: "฿", thb: "฿",
  shekel: "₪", shekels: "₪", ils: "₪",
  naira: "₦", ngn: "₦",
  php: "₱",
  dong: "₫", vnd: "₫",
  rupiah: "Rp", idr: "Rp", ringgit: "RM", myr: "RM",
  hryvnia: "₴", uah: "₴",
  aed: "AED", sar: "SAR", qar: "QAR",
  btc: "₿", bitcoin: "₿",
};

/**
 * Currencies whose sign conventionally FOLLOWS the number, with a space: "100 kr", "100 zł", the
 * Gulf ISO codes. Everything else hugs the front of the number ("$100", "€100", "R$100").
 */
const SYMBOL_AFTER = new Set(["kr", "zł", "AED", "SAR", "QAR"]);

/** Which side `symbol` sits on and whether it takes a spacing gap. */
export function symbolPlacement(symbol: string): { before: boolean; space: boolean } {
  const after = SYMBOL_AFTER.has(symbol);
  return { before: !after, space: after };
}

/**
 * Symbols recognised by the "already has a symbol" reflow path, longest first so multi-character
 * signs ("R$") match before their single-character prefix. Kept to unambiguous currency marks - bare
 * letter clusters like "kr" are only ever produced by the word path, never scanned for here.
 */
const FORMAT_SYMBOLS = ["R$", "$", "€", "£", "¥", "₹", "₽", "₩", "₺", "₴", "₪", "₦", "₱", "₫", "฿", "₿", "Rp", "RM"];

/** The symbol a currency WORD/code maps to, or null. Exposed for the suggestion popup. */
export function currencySymbolForWord(word: string): string | null {
  return WORD_TO_SYMBOL[word.toLowerCase()] ?? null;
}

/** Is `prefix` the start of any currency word/code? Lets the popup keep triggering on a number-glued
 *  letter run ("1000eu…") that would otherwise be dismissed as an ordinal/unit suffix. */
export function isCurrencyWordPrefix(prefix: string): boolean {
  if (!prefix) return false;
  const p = prefix.toLowerCase();
  for (const key in WORD_TO_SYMBOL) if (key.startsWith(p)) return true;
  return false;
}

/** Group a pure-digit integer string into threes: "1000000" -> "1,000,000". */
export function groupThousands(digits: string, sep: string): string {
  let out = "";
  for (let i = 0; i < digits.length; i++) {
    if (i > 0 && (digits.length - i) % 3 === 0) out += sep;
    out += digits[i];
  }
  return out;
}

/**
 * Parse a raw typed amount into integer digits and an optional decimal part, remembering which
 * separator the user used for the decimal. A separator followed by 1-2 digits at the very END is the
 * decimal; any other separator is grouping and is stripped. Returns null when there are no digits.
 */
export function parseAmount(raw: string): { int: string; dec: string; decSep: string } | null {
  if (!/\d/.test(raw)) return null;
  let body = raw.trim();
  let dec = "";
  let decSep = ".";
  const decMatch = body.match(/([.,])(\d{1,2})$/);
  // Treat a trailing "sep + 1-2 digits" as a decimal only if digits precede it, so "1,000" (a group
  // of three) stays an integer while "1,50" reads as 1.50.
  if (decMatch && /\d/.test(body.slice(0, body.length - decMatch[0].length))) {
    decSep = decMatch[1];
    dec = decMatch[2];
    body = body.slice(0, body.length - decMatch[0].length);
  }
  const int = body.replace(/\D/g, "");
  if (!int) return null;
  return { int, dec, decSep };
}

/**
 * Reformat a raw amount's digits with the style's thousands separator, keeping (or converting) the
 * decimal mark. With a "none" style the decimal the user typed is preserved; otherwise it becomes
 * the style's decimal (the one the thousands separator does not use).
 */
export function formatAmount(raw: string, style: CurrencyStyle): string {
  const p = parseAmount(raw);
  if (!p) return raw;
  const grouped = groupThousands(p.int, style.thousands);
  if (!p.dec) return grouped;
  return grouped + (style.decimal ?? p.decSep) + p.dec;
}

/** Assemble a symbol and a formatted number on the currency's conventional side. */
export function composeCurrency(symbol: string, amount: string): string {
  const { before, space } = symbolPlacement(symbol);
  return before ? `${symbol}${amount}` : `${amount}${space ? " " : ""}${symbol}`;
}

export interface CurrencyOptions {
  /** Reflow an amount that already carries a symbol. */
  format: boolean;
  /** Replace a currency WORD after a number with its symbol. */
  wordToSymbol: boolean;
  style: CurrencyStyle;
}

/**
 * Look at the text just before the caret (any boundary character already removed) and, if it ends in
 * a currency expression, return the slice to replace and its replacement. `start` is the index in
 * `before` where the replacement begins; the caller replaces `before.slice(start)`.
 *
 * The number is a single contiguous run of digits and grouping marks - NOT spanning a space - so
 * "2003 34 dollars" converts only "34 dollars" (to "2003 $34") and "¥1,000 1000 yuan" only touches
 * the trailing "1000 yuan". Returns null when nothing applies, the result equals the input, or the
 * number is glued to a letter (so "abc1000 dollars" is left alone).
 */
export function detectCurrency(before: string, opts: CurrencyOptions): { start: number; text: string } | null {
  // Case A - number then a spelled-out currency word: "1000 dollars" -> "$1,000". `\s*` allows zero
  // spaces, so "1000euro" works too.
  if (opts.wordToSymbol) {
    const m = before.match(/(\d[\d.,]*)\s*([A-Za-z]+)$/);
    if (m) {
      const symbol = WORD_TO_SYMBOL[m[2].toLowerCase()];
      const start = before.length - m[0].length;
      if (symbol && parseAmount(m[1]) && boundaryOk(before, start)) {
        return { start, text: composeCurrency(symbol, formatAmount(m[1], opts.style)) };
      }
    }
  }
  // Case B - an amount that already has a symbol, on either side: "$1000", "1000$", "$ 1000". The
  // symbol is moved to its conventional side by composeCurrency.
  if (opts.format) {
    const sym = `(?:${FORMAT_SYMBOLS.map(escapeRe).join("|")})`;
    let m = before.match(new RegExp(`(${sym})\\s?(\\d[\\d.,]*)$`)); // symbol first
    if (m && parseAmount(m[2])) {
      const start = before.length - m[0].length;
      if (boundaryOk(before, start)) {
        const text = composeCurrency(m[1], formatAmount(m[2], opts.style));
        return text === before.slice(start) ? null : { start, text };
      }
    }
    m = before.match(new RegExp(`(\\d[\\d.,]*)\\s?(${sym})$`)); // symbol last
    if (m && parseAmount(m[1])) {
      const start = before.length - m[0].length;
      if (boundaryOk(before, start)) {
        const text = composeCurrency(m[2], formatAmount(m[1], opts.style));
        return text === before.slice(start) ? null : { start, text };
      }
    }
  }
  return null;
}

/** The char before the expression must not be a letter/digit/dot (so we don't slice mid-token). */
function boundaryOk(before: string, start: number): boolean {
  if (start <= 0) return true;
  return !/[\w.]/.test(before[start - 1]);
}

/** Escape a symbol for safe insertion into a RegExp alternation. */
function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
