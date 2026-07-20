/**
 * Folder/file exclusion matching for "don't run in these places".
 *
 * A pattern matches a vault-relative file path (POSIX "/" separators) if it is:
 *   - an exact path            "Templates/daily.md"
 *   - a folder prefix          "Templates"  or  "Templates/"  → any file beneath it
 *   - a glob with * and ?      "*.excalidraw.md", "Journal/**", "**\/private/**"
 *
 * `*` matches within a path segment, `**` matches across segments, `?` one char.
 * Matching is case-insensitive (Obsidian paths usually are on macOS/Windows) and
 * anchored to the whole path. Blank patterns are ignored.
 */
function globToRegExp(pattern: string): RegExp {
  let re = "";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === "*") {
      if (pattern[i + 1] === "*") {
        re += ".*"; // ** - across segments
        i++;
        if (pattern[i + 1] === "/") i++; // swallow the slash so "**/x" also matches "x"
      } else {
        re += "[^/]*"; // * - within a segment
      }
    } else if (c === "?") {
      re += "[^/]";
    } else {
      re += c.replace(/[.+^${}()|[\]\\]/g, "\\$&");
    }
  }
  return new RegExp(`^${re}$`, "i");
}

/** True if `path` is excluded by any of the patterns. */
export function pathExcluded(path: string, patterns: readonly string[]): boolean {
  const p = path.replace(/^\/+/, "");
  for (const raw of patterns) {
    let pat = raw.trim().replace(/^\/+/, "").replace(/\/+$/, "");
    if (!pat) continue;
    if (pat.includes("*") || pat.includes("?")) {
      if (globToRegExp(pat).test(p)) return true;
    } else {
      // exact file, or a folder prefix (everything beneath it)
      if (p.toLowerCase() === pat.toLowerCase()) return true;
      if (p.toLowerCase().startsWith(pat.toLowerCase() + "/")) return true;
    }
  }
  return false;
}

/** Parse a user's textarea (newline- or comma-separated) into clean patterns. */
export function parseExcludeList(text: string): string[] {
  return text
    .split(/[\n,]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}
