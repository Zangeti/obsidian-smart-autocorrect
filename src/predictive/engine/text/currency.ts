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
  /** Place the euro sign AFTER the number ("100 €") instead of before ("€100"). */
  euroAfter?: boolean;
  /** Emit the ISO code after the number ("1,000 USD") instead of the symbol ("$1,000"). */
  useCode?: boolean;
}

/** Map the user settings to a full style. */
export function currencyStyleFor(
  thousands: "comma" | "period" | "none",
  opts: { euroAfter?: boolean; useCode?: boolean } = {},
): CurrencyStyle {
  const base =
    thousands === "period"
      ? { thousands: ".", decimal: "," } //  1.000,50
      : thousands === "none"
        ? { thousands: "", decimal: null } // 1000.50 or 1000,50 (kept)
        : { thousands: ",", decimal: "." }; // 1,000.50
  return { ...base, euroAfter: opts.euroAfter, useCode: opts.useCode };
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
 * Currencies whose sign conventionally FOLLOWS the number: the Nordic krona, Polish złoty, Swiss
 * franc, Vietnamese dong, Ukrainian hryvnia, Russian ruble, and the Gulf ISO codes. Everything else
 * hugs the front of the number ("$100", "€100", "₹100", "¥100", "R$100").
 */
const SYMBOL_AFTER = new Set(["kr", "zł", "Fr", "₫", "₴", "₽", "AED", "SAR", "QAR"]);

/** Spelled-out currency words / codes → ISO 4217 code, for the "use code instead of symbol" mode.
 *  Distinguishes currencies that share a sign ($ → USD but "cad" → CAD, ¥ → JPY but "yuan" → CNY). */
const WORD_TO_CODE: Record<string, string> = {
  dollar: "USD", dollars: "USD", usd: "USD", buck: "USD", bucks: "USD",
  cad: "CAD", aud: "AUD", nzd: "NZD", sgd: "SGD", hkd: "HKD", mxn: "MXN", peso: "MXN", pesos: "MXN",
  euro: "EUR", euros: "EUR", eur: "EUR",
  pound: "GBP", pounds: "GBP", gbp: "GBP", quid: "GBP", sterling: "GBP",
  yen: "JPY", jpy: "JPY", yuan: "CNY", cny: "CNY", rmb: "CNY", renminbi: "CNY",
  rupee: "INR", rupees: "INR", inr: "INR",
  franc: "CHF", francs: "CHF", chf: "CHF",
  ruble: "RUB", rubles: "RUB", rouble: "RUB", roubles: "RUB", rub: "RUB",
  won: "KRW", krw: "KRW",
  lira: "TRY", try: "TRY",
  real: "BRL", reais: "BRL", brl: "BRL",
  rand: "ZAR", zar: "ZAR",
  krona: "SEK", kronor: "SEK", krone: "NOK", kroner: "DKK", sek: "SEK", nok: "NOK", dkk: "DKK",
  zloty: "PLN", zlotych: "PLN", pln: "PLN",
  baht: "THB", thb: "THB",
  shekel: "ILS", shekels: "ILS", ils: "ILS",
  naira: "NGN", ngn: "NGN",
  php: "PHP",
  dong: "VND", vnd: "VND",
  rupiah: "IDR", idr: "IDR", ringgit: "MYR", myr: "MYR",
  hryvnia: "UAH", uah: "UAH",
  aed: "AED", sar: "SAR", qar: "QAR",
  btc: "BTC", bitcoin: "BTC",
};

/** Default ISO code for a bare symbol (a "$" amount with no word is assumed USD, etc.). */
const SYMBOL_TO_CODE: Record<string, string> = {
  "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR", "₽": "RUB", "₩": "KRW", "₺": "TRY",
  "₴": "UAH", "₪": "ILS", "₦": "NGN", "₱": "PHP", "₫": "VND", "฿": "THB", "₿": "BTC", "R$": "BRL",
  "Rp": "IDR", "RM": "MYR", "R": "ZAR", "kr": "SEK", "zł": "PLN", "Fr": "CHF",
};

/** Which side `symbol` sits on and whether it takes a spacing gap. The euro follows the style
 *  setting (before by default); every other currency follows its fixed convention. */
export function symbolPlacement(symbol: string, style?: CurrencyStyle): { before: boolean; space: boolean } {
  if (symbol === "€") return { before: !style?.euroAfter, space: !!style?.euroAfter };
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
 * Split a raw typed amount into integer digits and decimal digits, using the STYLE to decide which
 * separator is the decimal (the one the thousands separator does not use), so a 3-digit decimal like
 * "10.567" is not mistaken for a thousands group. With a "none" style there is no thousands
 * separator, so the last "." or "," the user typed is the decimal. Returns null when no digits.
 */
export function parseAmount(raw: string, style: CurrencyStyle = { thousands: ",", decimal: "." }): { int: string; dec: string; decSep: string } | null {
  if (!/\d/.test(raw)) return null;
  let body = raw.trim();
  // Which character marks the decimal here?
  let decChar: string | null;
  if (style.thousands === ",") decChar = ".";
  else if (style.thousands === ".") decChar = ",";
  else {
    const lastDot = body.lastIndexOf("."), lastComma = body.lastIndexOf(",");
    decChar = lastDot < 0 && lastComma < 0 ? null : lastDot > lastComma ? "." : ",";
  }
  let dec = "";
  let decSep = decChar ?? ".";
  if (decChar) {
    const idx = body.lastIndexOf(decChar);
    const after = idx >= 0 ? body.slice(idx + 1) : "";
    // A decimal is the LAST occurrence of the decimal char followed by pure digits, with digits
    // before it. Anything else (a stray thousands mark) is stripped as grouping.
    if (idx >= 0 && /^\d+$/.test(after) && /\d/.test(body.slice(0, idx))) {
      dec = after;
      body = body.slice(0, idx);
    }
  }
  const int = body.replace(/\D/g, "");
  if (!int) return null;
  return { int, dec, decSep };
}

/**
 * Reformat a raw amount with the style's thousands separator and decimal mark. When there is a
 * decimal part it is padded to AT LEAST two places (a lone "1000.5" becomes "1,000.50"); more places
 * are kept as typed ("10.567" stays three).
 */
export function formatAmount(raw: string, style: CurrencyStyle): string {
  const p = parseAmount(raw, style);
  if (!p) return raw;
  const grouped = groupThousands(p.int, style.thousands);
  if (!p.dec) return grouped;
  const dec = p.dec.length < 2 ? p.dec.padEnd(2, "0") : p.dec;
  return grouped + (style.decimal ?? p.decSep) + dec;
}

/** Assemble a symbol and a formatted number on the currency's conventional side (euro follows the
 *  style setting). Does not handle the sign or ISO-code mode - see {@link buildCurrency}. */
export function composeCurrency(symbol: string, amount: string, style?: CurrencyStyle): string {
  const { before, space } = symbolPlacement(symbol, style);
  return before ? `${symbol}${amount}` : `${amount}${space ? " " : ""}${symbol}`;
}

/**
 * Turn an identified currency plus a raw number into the final surface, honouring the sign, the
 * ISO-code mode ("1,000 USD"), the euro placement, and the thousands/decimal style. `sourceWord` is
 * the word/code the user typed, when known, so a shared sign resolves to the right code ($ but CAD).
 */
function buildCurrency(symbol: string, rawNumber: string, style: CurrencyStyle, sourceWord?: string): string {
  const sign = /^\s*-/.test(rawNumber) ? "-" : "";
  const amount = formatAmount(rawNumber.replace(/^\s*-/, ""), style);
  if (style.useCode) {
    const code = (sourceWord && WORD_TO_CODE[sourceWord.toLowerCase()]) || SYMBOL_TO_CODE[symbol] || symbol;
    return `${sign}${amount} ${code}`;
  }
  return sign + composeCurrency(symbol, amount, style);
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
  // spaces, so "1000euro" works too; an optional leading "-" gives a negative amount.
  if (opts.wordToSymbol) {
    const m = before.match(/(-?\d[\d.,]*)\s*([A-Za-z]+)$/);
    if (m) {
      const symbol = WORD_TO_SYMBOL[m[2].toLowerCase()];
      const start = before.length - m[0].length;
      if (symbol && parseAmount(m[1], opts.style) && boundaryOk(before, start)) {
        return { start, text: buildCurrency(symbol, m[1], opts.style, m[2]) };
      }
    }
  }
  // Case B - an amount that already has a symbol, on either side: "$1000", "1000$", "-$1000". The
  // symbol/code is moved to its conventional side by buildCurrency.
  if (opts.format) {
    const sym = `(?:${FORMAT_SYMBOLS.map(escapeRe).join("|")})`;
    let m = before.match(new RegExp(`(-?)(${sym})\\s?(\\d[\\d.,]*)$`)); // symbol first (opt. sign)
    if (m && parseAmount(m[3], opts.style)) {
      const start = before.length - m[0].length;
      if (boundaryOk(before, start)) {
        const text = buildCurrency(m[2], m[1] + m[3], opts.style);
        return text === before.slice(start) ? null : { start, text };
      }
    }
    m = before.match(new RegExp(`(-?\\d[\\d.,]*)\\s?(${sym})$`)); // symbol last (opt. sign)
    if (m && parseAmount(m[1], opts.style)) {
      const start = before.length - m[0].length;
      if (boundaryOk(before, start)) {
        const text = buildCurrency(m[2], m[1], opts.style);
        return text === before.slice(start) ? null : { start, text };
      }
    }
  }
  return null;
}

/** The symbol of the first currency word/code that `prefix` starts (for the partial-word popup). */
function firstSymbolForPrefix(prefix: string): string | null {
  const p = prefix.toLowerCase();
  for (const key in WORD_TO_SYMBOL) if (key.startsWith(p)) return WORD_TO_SYMBOL[key];
  return null;
}

/**
 * A currency proposal for the suggestion POPUP, from the text up to the caret. Unlike
 * {@link detectCurrency} it also fires on a PARTIAL currency word ("1000 doll…"), resolving it to
 * the currency it is heading toward, so the popup can offer a single "format currency" action
 * (with the target symbol) instead of autocompleting the word. Returns the span to replace, the
 * formatted result, and the symbol.
 */
export function currencyProposal(before: string, opts: CurrencyOptions): { start: number; text: string; symbol: string } | null {
  const badgeOf = (text: string) => text.replace(/[-\d.,\s]/g, ""); // the symbol/code, minus the number
  const full = detectCurrency(before, opts);
  if (full) return { start: full.start, text: full.text, symbol: badgeOf(full.text) };
  if (opts.wordToSymbol) {
    const m = before.match(/(-?\d[\d.,]*)\s*([A-Za-z]+)$/);
    if (m && !WORD_TO_SYMBOL[m[2].toLowerCase()] && isCurrencyWordPrefix(m[2])) {
      const symbol = firstSymbolForPrefix(m[2]);
      const start = before.length - m[0].length;
      if (symbol && parseAmount(m[1], opts.style) && boundaryOk(before, start)) {
        const text = buildCurrency(symbol, m[1], opts.style);
        return { start, text, symbol: badgeOf(text) };
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
