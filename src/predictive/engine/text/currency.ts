/**
 * Currency tidying: turn "$1000" into "$1,000" (or "1.000 €"), and "1000 dollars" into the right
 * symbol with the number reformatted. Pure and unit-tested; the plugin runs it on a word boundary.
 *
 * Two independent behaviours, each gated by the caller:
 *   - format:       reflow an amount that already has a symbol (group thousands, put the symbol on
 *                   the configured side).
 *   - wordToSymbol: replace a spelled-out currency word after a number with its symbol.
 *
 * A locale "style" decides the thousands/decimal separators and where the symbol sits, so an
 * English writer gets "$1,000.50" and a German one "1.000,50 €" from the same input.
 */

export interface CurrencyStyle {
  /** Groups-of-three separator, e.g. "," (English) or "." (German). */
  thousands: string;
  /** Decimal point, e.g. "." (English) or "," (German). */
  decimal: string;
  /** Symbol before the number ("$1,000") vs after it ("1.000 €"). */
  symbolBefore: boolean;
  /** A space between the symbol and the number ("1.000 €" vs "$1000"). */
  space: boolean;
}

/** The two presets the settings expose. `thousands`/`decimal`/side all move together per locale. */
export const CURRENCY_STYLES: Record<"english" | "german", CurrencyStyle> = {
  english: { thousands: ",", decimal: ".", symbolBefore: true, space: false },
  german: { thousands: ".", decimal: ",", symbolBefore: false, space: true },
};

/**
 * Build a style from the user's chosen thousands separator. Comma and "none" follow the English
 * convention (decimal point, symbol before the number); a period separator follows the European one
 * (decimal comma, symbol after with a space), which is the only combination in which a "." thousands
 * mark is unambiguous.
 */
export function currencyStyleFor(thousands: "comma" | "period" | "none"): CurrencyStyle {
  if (thousands === "period") return CURRENCY_STYLES.german;
  if (thousands === "none") return { thousands: "", decimal: ".", symbolBefore: true, space: false };
  return CURRENCY_STYLES.english;
}

/**
 * Spelled-out currency words / ISO codes → symbol. Lower-cased keys, matched case-insensitively, so
 * "USD", "usd", "Dollars" all resolve. Covers the major world currencies; several legitimately
 * share a sign ($ for USD/CAD/AUD/NZD/SGD/HKD/MXN, ¥ for JPY/CNY, kr for the Nordic krona/krone),
 * which is fine - the sign is what the writer wants on the page.
 */
const WORD_TO_SYMBOL: Record<string, string> = {
  // US dollar and the other dollar currencies (all use "$").
  dollar: "$", dollars: "$", usd: "$", buck: "$", bucks: "$",
  cad: "$", aud: "$", nzd: "$", sgd: "$", hkd: "$", mxn: "$", peso: "$", pesos: "$",
  // Euro.
  euro: "€", euros: "€", eur: "€",
  // Pound sterling.
  pound: "£", pounds: "£", gbp: "£", quid: "£", sterling: "£",
  // Yen / yuan / renminbi.
  yen: "¥", jpy: "¥", yuan: "¥", cny: "¥", rmb: "¥", renminbi: "¥",
  // Indian rupee (and other rupees share the sign closely enough).
  rupee: "₹", rupees: "₹", inr: "₹",
  // Swiss franc / CFA franc.
  franc: "Fr", francs: "Fr", chf: "Fr",
  // Russian rouble.
  ruble: "₽", rubles: "₽", rouble: "₽", roubles: "₽", rub: "₽",
  // Korean won.
  won: "₩", krw: "₩",
  // Turkish lira.
  lira: "₺", try: "₺",
  // Brazilian real.
  real: "R$", reais: "R$", brl: "R$",
  // South African rand.
  rand: "R", zar: "R",
  // Nordic krona / krone.
  krona: "kr", kronor: "kr", krone: "kr", kroner: "kr", sek: "kr", nok: "kr", dkk: "kr",
  // Polish złoty.
  zloty: "zł", zlotych: "zł", pln: "zł",
  // Thai baht.
  baht: "฿", thb: "฿",
  // Israeli shekel.
  shekel: "₪", shekels: "₪", ils: "₪",
  // Nigerian naira.
  naira: "₦", ngn: "₦",
  // Philippine peso (distinct sign from the Latin-American pesos above).
  php: "₱",
  // Vietnamese dong.
  dong: "₫", vnd: "₫",
  // Indonesian rupiah / Malaysian ringgit.
  rupiah: "Rp", idr: "Rp", ringgit: "RM", myr: "RM",
  // Ukrainian hryvnia.
  hryvnia: "₴", uah: "₴",
  // Gulf & other common ISO codes (no distinct Unicode sign in wide use → keep the code).
  aed: "AED", sar: "SAR", qar: "QAR",
  // Crypto, since people write it the same way.
  btc: "₿", bitcoin: "₿",
};

/**
 * Symbols recognised by the "already has a symbol" reflow path, longest first so multi-character
 * signs ("R$") match before their single-character prefix. Kept to unambiguous currency marks -
 * bare letter clusters like "kr" are only produced by the word path, never scanned for here.
 */
const FORMAT_SYMBOLS = ["R$", "$", "€", "£", "¥", "₹", "₽", "₩", "₺", "₴", "₪", "₦", "₱", "₫", "฿", "₿"];

/** Group a pure-digit integer string into threes: "1000000" → "1,000,000". */
export function groupThousands(digits: string, sep: string): string {
  let out = "";
  for (let i = 0; i < digits.length; i++) {
    if (i > 0 && (digits.length - i) % 3 === 0) out += sep;
    out += digits[i];
  }
  return out;
}

/**
 * Parse a raw typed amount (which may contain grouping separators, spaces, and one decimal part)
 * into pure integer digits and up-to-two decimal digits. A separator immediately followed by 1-2
 * digits at the very END is read as the decimal; every other separator or space is grouping and is
 * stripped. Returns null when there are no digits.
 */
export function parseAmount(raw: string): { int: string; dec: string } | null {
  if (!/\d/.test(raw)) return null;
  let body = raw.trim();
  let dec = "";
  const decMatch = body.match(/[.,](\d{1,2})$/);
  // Only treat a trailing separator group as a decimal if what precedes it also has digits, so a
  // bare "1,00" reads as 1.00 but a lone grouping like "1,000" (3 digits) stays integer.
  if (decMatch && decMatch[1].length <= 2 && /\d/.test(body.slice(0, body.length - decMatch[0].length))) {
    dec = decMatch[1];
    body = body.slice(0, body.length - decMatch[0].length);
  }
  const int = body.replace(/\D/g, "");
  if (!int) return null;
  return { int, dec };
}

/** Reformat a raw amount's digits with the style's separators. "1000000" → "1,000,000". */
export function formatAmount(raw: string, style: CurrencyStyle): string {
  const p = parseAmount(raw);
  if (!p) return raw;
  const grouped = groupThousands(p.int, style.thousands);
  return p.dec ? grouped + style.decimal + p.dec : grouped;
}

/** Assemble a symbol and a formatted number per the style's side/spacing. */
export function composeCurrency(symbol: string, amount: string, style: CurrencyStyle): string {
  const gap = style.space ? " " : "";
  return style.symbolBefore ? `${symbol}${gap}${amount}` : `${amount}${gap}${symbol}`;
}

export interface CurrencyOptions {
  /** Reflow an amount that already carries a symbol. */
  format: boolean;
  /** Replace a currency WORD after a number with its symbol. */
  wordToSymbol: boolean;
  style: CurrencyStyle;
}

/**
 * Look at the text just before the caret (the boundary character already removed) and, if it ends
 * in a currency expression, return the slice to replace and its replacement. `start` is the index
 * in `before` where the replacement begins; the caller replaces `before.slice(start)`.
 *
 * Returns null when nothing applies, when the result would equal the input, or when the number is
 * glued to a letter/word (so "css3000px" or a hex-ish token is never touched).
 */
export function detectCurrency(before: string, opts: CurrencyOptions): { start: number; text: string } | null {
  // Case A — number then a spelled-out currency word: "1000 dollars" → "$1,000".
  if (opts.wordToSymbol) {
    const m = before.match(/(\d[\d., ]*?)\s*([A-Za-z]+)$/);
    if (m) {
      const symbol = WORD_TO_SYMBOL[m[2].toLowerCase()];
      const numRaw = m[1].trim();
      const start = before.length - m[0].length + (m[0].length - m[0].trimStart().length);
      if (symbol && parseAmount(numRaw) && boundaryOk(before, start)) {
        return { start, text: composeCurrency(symbol, formatAmount(numRaw, opts.style), opts.style) };
      }
    }
  }
  // Case B — an amount that already has a symbol, on either side: "$1000", "1000$", "$ 1000".
  if (opts.format) {
    const sym = `(?:${FORMAT_SYMBOLS.map(escapeRe).join("|")})`;
    let m = before.match(new RegExp(`(${sym})\\s?(\\d[\\d., ]*?)$`)); // symbol first
    if (m && parseAmount(m[2])) {
      const start = before.length - m[0].length;
      if (boundaryOk(before, start)) {
        const text = composeCurrency(m[1], formatAmount(m[2].trim(), opts.style), opts.style);
        return text === before.slice(start) ? null : { start, text };
      }
    }
    m = before.match(new RegExp(`(\\d[\\d., ]*?)\\s?(${sym})$`)); // symbol last
    if (m && parseAmount(m[1])) {
      const start = before.length - m[0].length;
      if (boundaryOk(before, start)) {
        const text = composeCurrency(m[2], formatAmount(m[1].trim(), opts.style), opts.style);
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
