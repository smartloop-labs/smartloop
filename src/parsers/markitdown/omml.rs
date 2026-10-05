//! Office Math Markup Language (OMML) → LaTeX: hand-port of MarkItDown's
//! `converter_utils/docx/math/omml.py` and `latex_dict.py` (adapted from
//! <https://github.com/xiilei/dwml>).
//!
//! The Python converter dispatches on the local tag name of every `m:`
//! child and formats its LaTeX templates with `str.format`; templates and
//! error behaviour (missing children raise, unsupported functions raise) are
//! reproduced exactly, since an error makes the DOCX pre-processor keep the
//! original XML part.

use std::collections::HashMap;

use super::ooxml::{XmlElement, XmlNode};

/// Characters escaped with a backslash by [`escape_latex`].
const CHARS: &[char] = &['{', '}', '_', '^', '#', '&', '$', '%', '~'];

const BLANK: &str = "";
const BACKSLASH: char = '\\';
const ALN: &str = "&";
const BRK: &str = "\\\\";
const FUNC_PLACE: &str = "{fe}";

/// Accents and group characters (`CHR`): unicode -> LaTeX template.
const CHR: &[(&str, &str)] = &[
    // Top accents
    ("\u{0300}", "\\grave{{{0}}}"),
    ("\u{0301}", "\\acute{{{0}}}"),
    ("\u{0302}", "\\hat{{{0}}}"),
    ("\u{0303}", "\\tilde{{{0}}}"),
    ("\u{0304}", "\\bar{{{0}}}"),
    ("\u{0305}", "\\overbar{{{0}}}"),
    ("\u{0306}", "\\breve{{{0}}}"),
    ("\u{0307}", "\\dot{{{0}}}"),
    ("\u{0308}", "\\ddot{{{0}}}"),
    ("\u{0309}", "\\ovhook{{{0}}}"),
    ("\u{030a}", "\\ocirc{{{0}}}}"),
    ("\u{030c}", "\\check{{{0}}}}"),
    ("\u{0310}", "\\candra{{{0}}}"),
    ("\u{0312}", "\\oturnedcomma{{{0}}}"),
    ("\u{0315}", "\\ocommatopright{{{0}}}"),
    ("\u{031a}", "\\droang{{{0}}}"),
    ("\u{0338}", "\\not{{{0}}}"),
    ("\u{20d0}", "\\leftharpoonaccent{{{0}}}"),
    ("\u{20d1}", "\\rightharpoonaccent{{{0}}}"),
    ("\u{20d2}", "\\vertoverlay{{{0}}}"),
    ("\u{20d6}", "\\overleftarrow{{{0}}}"),
    ("\u{20d7}", "\\vec{{{0}}}"),
    ("\u{20db}", "\\dddot{{{0}}}"),
    ("\u{20dc}", "\\ddddot{{{0}}}"),
    ("\u{20e1}", "\\overleftrightarrow{{{0}}}"),
    ("\u{20e7}", "\\annuity{{{0}}}"),
    ("\u{20e9}", "\\widebridgeabove{{{0}}}"),
    ("\u{20f0}", "\\asteraccent{{{0}}}"),
    // Bottom accents
    ("\u{0330}", "\\wideutilde{{{0}}}"),
    ("\u{0331}", "\\underbar{{{0}}}"),
    ("\u{20e8}", "\\threeunderdot{{{0}}}"),
    ("\u{20ec}", "\\underrightharpoondown{{{0}}}"),
    ("\u{20ed}", "\\underleftharpoondown{{{0}}}"),
    ("\u{20ee}", "\\underledtarrow{{{0}}}"),
    ("\u{20ef}", "\\underrightarrow{{{0}}}"),
    // Over | group
    ("\u{23b4}", "\\overbracket{{{0}}}"),
    ("\u{23dc}", "\\overparen{{{0}}}"),
    ("\u{23de}", "\\overbrace{{{0}}}"),
    // Under| group
    ("\u{23b5}", "\\underbracket{{{0}}}"),
    ("\u{23dd}", "\\underparen{{{0}}}"),
    ("\u{23df}", "\\underbrace{{{0}}}"),
];

/// Big operators (`CHR_BO`).
const CHR_BO: &[(&str, &str)] = &[
    ("\u{2140}", "\\Bbbsum"),
    ("\u{220f}", "\\prod"),
    ("\u{2210}", "\\coprod"),
    ("\u{2211}", "\\sum"),
    ("\u{222b}", "\\int"),
    ("\u{22c0}", "\\bigwedge"),
    ("\u{22c1}", "\\bigvee"),
    ("\u{22c2}", "\\bigcap"),
    ("\u{22c3}", "\\bigcup"),
    ("\u{2a00}", "\\bigodot"),
    ("\u{2a01}", "\\bigoplus"),
    ("\u{2a02}", "\\bigotimes"),
];

/// Text symbols (`T`): unicode character -> LaTeX.
const T: &[(&str, &str)] = &[
    // Greek letters
    ("\u{1d6fc}", "\\alpha "),
    ("\u{1d6fd}", "\\beta "),
    ("\u{1d6fe}", "\\gamma "),
    ("\u{1d6ff}", "\\theta "),
    ("\u{1d700}", "\\epsilon "),
    ("\u{1d701}", "\\zeta "),
    ("\u{1d702}", "\\eta "),
    ("\u{1d703}", "\\theta "),
    ("\u{1d704}", "\\iota "),
    ("\u{1d705}", "\\kappa "),
    ("\u{1d706}", "\\lambda "),
    ("\u{1d707}", "\\m "),
    ("\u{1d708}", "\\n "),
    ("\u{1d709}", "\\xi "),
    ("\u{1d70a}", "\\omicron "),
    ("\u{1d70b}", "\\pi "),
    ("\u{1d70c}", "\\rho "),
    ("\u{1d70d}", "\\varsigma "),
    ("\u{1d70e}", "\\sigma "),
    ("\u{1d70f}", "\\ta "),
    ("\u{1d710}", "\\upsilon "),
    ("\u{1d711}", "\\phi "),
    ("\u{1d712}", "\\chi "),
    ("\u{1d713}", "\\psi "),
    ("\u{1d714}", "\\omega "),
    ("\u{1d715}", "\\partial "),
    ("\u{1d716}", "\\varepsilon "),
    ("\u{1d717}", "\\vartheta "),
    ("\u{1d718}", "\\varkappa "),
    ("\u{1d719}", "\\varphi "),
    ("\u{1d71a}", "\\varrho "),
    ("\u{1d71b}", "\\varpi "),
    // Relation symbols
    ("\u{2190}", "\\leftarrow "),
    ("\u{2191}", "\\uparrow "),
    ("\u{2192}", "\\rightarrow "),
    ("\u{2193}", "\\downright "),
    ("\u{2194}", "\\leftrightarrow "),
    ("\u{2195}", "\\updownarrow "),
    ("\u{2196}", "\\nwarrow "),
    ("\u{2197}", "\\nearrow "),
    ("\u{2198}", "\\searrow "),
    ("\u{2199}", "\\swarrow "),
    ("\u{22ee}", "\\vdots "),
    ("\u{22ef}", "\\cdots "),
    ("\u{22f0}", "\\adots "),
    ("\u{22f1}", "\\ddots "),
    ("\u{2260}", "\\ne "),
    ("\u{2264}", "\\leq "),
    ("\u{2265}", "\\geq "),
    ("\u{2266}", "\\leqq "),
    ("\u{2267}", "\\geqq "),
    ("\u{2268}", "\\lneqq "),
    ("\u{2269}", "\\gneqq "),
    ("\u{226a}", "\\ll "),
    ("\u{226b}", "\\gg "),
    ("\u{2208}", "\\in "),
    ("\u{2209}", "\\notin "),
    ("\u{220b}", "\\ni "),
    ("\u{220c}", "\\nni "),
    // Ordinary symbols
    ("\u{221e}", "\\infty "),
    // Binary relations
    ("\u{00b1}", "\\pm "),
    ("\u{2213}", "\\mp "),
    // Italic, Latin, uppercase
    ("\u{1d434}", "A"),
    ("\u{1d435}", "B"),
    ("\u{1d436}", "C"),
    ("\u{1d437}", "D"),
    ("\u{1d438}", "E"),
    ("\u{1d439}", "F"),
    ("\u{1d43a}", "G"),
    ("\u{1d43b}", "H"),
    ("\u{1d43c}", "I"),
    ("\u{1d43d}", "J"),
    ("\u{1d43e}", "K"),
    ("\u{1d43f}", "L"),
    ("\u{1d440}", "M"),
    ("\u{1d441}", "N"),
    ("\u{1d442}", "O"),
    ("\u{1d443}", "P"),
    ("\u{1d444}", "Q"),
    ("\u{1d445}", "R"),
    ("\u{1d446}", "S"),
    ("\u{1d447}", "T"),
    ("\u{1d448}", "U"),
    ("\u{1d449}", "V"),
    ("\u{1d44a}", "W"),
    ("\u{1d44b}", "X"),
    ("\u{1d44c}", "Y"),
    ("\u{1d44d}", "Z"),
    // Italic, Latin, lowercase
    ("\u{1d44e}", "a"),
    ("\u{1d44f}", "b"),
    ("\u{1d450}", "c"),
    ("\u{1d451}", "d"),
    ("\u{1d452}", "e"),
    ("\u{1d453}", "f"),
    ("\u{1d454}", "g"),
    ("\u{1d456}", "i"),
    ("\u{1d457}", "j"),
    ("\u{1d458}", "k"),
    ("\u{1d459}", "l"),
    ("\u{1d45a}", "m"),
    ("\u{1d45b}", "n"),
    ("\u{1d45c}", "o"),
    ("\u{1d45d}", "p"),
    ("\u{1d45e}", "q"),
    ("\u{1d45f}", "r"),
    ("\u{1d460}", "s"),
    ("\u{1d461}", "t"),
    ("\u{1d462}", "u"),
    ("\u{1d463}", "v"),
    ("\u{1d464}", "w"),
    ("\u{1d465}", "x"),
    ("\u{1d466}", "y"),
    ("\u{1d467}", "z"),
];

/// Function names (`FUNC`).
const FUNC: &[(&str, &str)] = &[
    ("sin", "\\sin({fe})"),
    ("cos", "\\cos({fe})"),
    ("tan", "\\tan({fe})"),
    ("arcsin", "\\arcsin({fe})"),
    ("arccos", "\\arccos({fe})"),
    ("arctan", "\\arctan({fe})"),
    ("arccot", "\\arccot({fe})"),
    ("sinh", "\\sinh({fe})"),
    ("cosh", "\\cosh({fe})"),
    ("tanh", "\\tanh({fe})"),
    ("coth", "\\coth({fe})"),
    ("sec", "\\sec({fe})"),
    ("csc", "\\csc({fe})"),
];

const CHR_DEFAULT_ACC_VAL: &str = "\\hat{{{0}}}";

const POS: &[(&str, &str)] = &[
    // not sure
    ("top", "\\overline{{{0}}}"),
    ("bot", "\\underline{{{0}}}"),
];

const POS_DEFAULT_BAR_VAL: &str = "\\overline{{{0}}}";

const SUB: &str = "_{{{0}}}";
const SUP: &str = "^{{{0}}}";

const F: &[(&str, &str)] = &[
    ("bar", "\\frac{{{num}}}{{{den}}}"),
    ("skw", "^{{{num}}}/_{{{den}}}"),
    ("noBar", "\\genfrac{{}}{{}}{{0pt}}{{}}{{{num}}}{{{den}}}"),
    ("lin", "{{{num}}}/{{{den}}}"),
];
const F_DEFAULT: &str = "\\frac{{{num}}}{{{den}}}";

const D: &str = "\\left{left}{text}\\right{right}";
const D_DEFAULT_LEFT: &str = "(";
const D_DEFAULT_RIGHT: &str = ")";
const D_DEFAULT_NULL: &str = ".";

const RAD: &str = "\\sqrt[{deg}]{{{text}}}";
const RAD_DEFAULT: &str = "\\sqrt{{{text}}}";

const ARR: &str = "\\begin{{array}}{{c}}{text}\\end{{array}}";

const LIM_FUNC: &[(&str, &str)] = &[
    ("lim", "\\lim_{{{lim}}}"),
    ("max", "\\max_{{{lim}}}"),
    ("min", "\\min_{{{lim}}}"),
];

const LIM_TO: (&str, &str) = ("\\rightarrow", "\\to");

const LIM_UPP: &str = "\\overset{{{lim}}}{{{text}}}";

const M: &str = "\\begin{{matrix}}{text}\\end{{matrix}}";

fn lookup(table: &[(&str, &'static str)], key: &str) -> Option<&'static str> {
    table.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
}

/// A value passed to Python's `str.format` (`None` prints as `None`).
fn py_none_or(value: Option<&str>) -> String {
    value.map(str::to_string).unwrap_or_else(|| "None".to_string())
}

/// Minimal Python `str.format`: `{{`/`}}` escapes, positional (`{0}` or
/// automatic `{}`) and named fields. Raises (like Python) on a single `}`,
/// an unclosed `{`, or an unknown field.
pub fn py_format(template: &str, positional: &[String], named: &[(&str, String)]) -> anyhow::Result<String> {
    let mut out = String::new();
    let chars: Vec<char> = template.chars().collect();
    let mut i = 0;
    let mut auto_index = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '{' {
            if chars.get(i + 1) == Some(&'{') {
                out.push('{');
                i += 2;
                continue;
            }
            let Some(close) = chars[i + 1..].iter().position(|&ch| ch == '}') else {
                anyhow::bail!("Single '{{' encountered in format string");
            };
            let field: String = chars[i + 1..i + 1 + close].iter().collect();
            if field.contains('{') {
                anyhow::bail!("unexpected '{{' in field name");
            }
            // Conversion (`!r`) and format specs are not used by the templates.
            let name = field.split(['!', ':']).next().unwrap_or("");
            let value = if name.is_empty() {
                let v = positional
                    .get(auto_index)
                    .ok_or_else(|| anyhow::anyhow!("Replacement index {auto_index} out of range for positional args tuple"))?;
                auto_index += 1;
                v.clone()
            } else if let Ok(index) = name.parse::<usize>() {
                positional
                    .get(index)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("Replacement index {index} out of range for positional args tuple"))?
            } else {
                named
                    .iter()
                    .find(|(k, _)| *k == name)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| anyhow::anyhow!("'{name}'"))?
            };
            out.push_str(&value);
            i += close + 2;
        } else if c == '}' {
            if chars.get(i + 1) == Some(&'}') {
                out.push('}');
                i += 2;
                continue;
            }
            anyhow::bail!("Single '}}' encountered in format string");
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
}

/// Escape LaTeX special characters not already preceded by a backslash.
pub fn escape_latex(strs: &str) -> String {
    let strs = strs.replace("\\\\", "\\");
    let mut last: Option<char> = None;
    let mut new_chr = String::with_capacity(strs.len());
    for c in strs.chars() {
        if CHARS.contains(&c) && last != Some(BACKSLASH) {
            new_chr.push(BACKSLASH);
            new_chr.push(c);
        } else {
            new_chr.push(c);
        }
        last = Some(c);
    }
    new_chr
}

/// `get_val(key, default, store)`: the store's mapping for `key` (or `key`
/// itself), or `default` when `key` is `None`.
fn get_val(key: Option<&str>, default: Option<&str>, store: &[(&str, &'static str)]) -> Option<String> {
    match key {
        Some(k) => Some(lookup(store, k).unwrap_or(k).to_string()),
        None => default.map(str::to_string),
    }
}

/// Common properties of an element (Python `Pr`): the text of its `brk`
/// children plus the `m:val` of its `chr`/`pos`/`begChr`/`endChr`/`type`
/// children.
#[derive(Debug, Clone, Default)]
struct Pr {
    text: String,
    values: HashMap<String, Option<String>>,
}

const PR_VAL_TAGS: &[&str] = &["chr", "pos", "begChr", "endChr", "type"];

impl Pr {
    fn new(elm: &XmlElement) -> Self {
        let mut pr = Pr::default();
        let mut text = String::new();
        for child in omml_children(elm) {
            let stag = local_name(child);
            match stag {
                "brk" => {
                    pr.values.insert("brk".to_string(), Some(BRK.to_string()));
                    text.push_str(BRK);
                }
                s if PR_VAL_TAGS.contains(&s) => {
                    pr.values.insert(s.to_string(), child.attr("m:val").map(str::to_string));
                }
                _ => {}
            }
        }
        pr.text = text;
        pr
    }

    /// Attribute lookup (`pr.chr` etc.): `None` when unset or valueless.
    fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).and_then(|v| v.as_deref())
    }
}

/// A converted child: LaTeX text, or a properties object.
#[derive(Debug, Clone)]
enum Latex {
    Text(String),
    Pr(Pr),
}

impl Latex {
    /// Python `str()` of the value.
    fn as_str(&self) -> &str {
        match self {
            Latex::Text(s) => s,
            Latex::Pr(pr) => &pr.text,
        }
    }

    fn pr(&self) -> anyhow::Result<&Pr> {
        match self {
            Latex::Pr(pr) => Ok(pr),
            Latex::Text(_) => anyhow::bail!("'str' object has no attribute"),
        }
    }

    /// The value where Python requires a `str` (e.g. `"".join(...)`).
    fn require_str(&self) -> anyhow::Result<&str> {
        match self {
            Latex::Text(s) => Ok(s),
            Latex::Pr(_) => anyhow::bail!("sequence item: expected str instance, Pr found"),
        }
    }
}

/// Element children in the math namespace.
fn omml_children(elm: &XmlElement) -> impl Iterator<Item = &XmlElement> {
    elm.elements().filter(|e| e.name.starts_with("m:"))
}

fn local_name(elm: &XmlElement) -> &str {
    elm.name.rsplit(':').next().unwrap_or(&elm.name)
}

/// Tags whose LaTeX is simply that of their children.
const DIRECT_TAGS: &[&str] = &["box", "sSub", "sSup", "sSubSup", "num", "den", "deg", "e"];

/// Convert an `m:oMath` element to LaTeX (Python `oMath2Latex(element).latex`).
pub fn omath_to_latex(element: &XmlElement) -> anyhow::Result<String> {
    process_children(element, None)
}

/// `process_children_list`: convert each math child (optionally only the
/// tags in `include`), skipping children that produce nothing.
fn process_children_list<'a>(
    elm: &'a XmlElement,
    include: Option<&[&str]>,
) -> anyhow::Result<Vec<(&'a str, Latex, &'a XmlElement)>> {
    let mut out = Vec::new();
    for child in omml_children(elm) {
        let stag = local_name(child);
        if let Some(include) = include
            && !include.contains(&stag)
        {
            continue;
        }
        let t = match call_method(child, stag)? {
            Some(t) => Some(t),
            None => process_unknow(child, stag)?,
        };
        if let Some(t) = t {
            out.push((stag, t, child));
        }
    }
    Ok(out)
}

/// `process_children_dict`: tag -> converted value (the last one wins).
fn process_children_dict(elm: &XmlElement, include: Option<&[&str]>) -> anyhow::Result<HashMap<String, Latex>> {
    let mut latex_chars = HashMap::new();
    for (stag, t, _) in process_children_list(elm, include)? {
        latex_chars.insert(stag.to_string(), t);
    }
    Ok(latex_chars)
}

/// `process_children`: the joined LaTeX of every child.
fn process_children(elm: &XmlElement, include: Option<&[&str]>) -> anyhow::Result<String> {
    Ok(process_children_list(elm, include)?.iter().map(|(_, t, _)| t.as_str().to_string()).collect::<Vec<_>>().join(BLANK))
}

fn process_unknow(elm: &XmlElement, stag: &str) -> anyhow::Result<Option<Latex>> {
    if DIRECT_TAGS.contains(&stag) {
        Ok(Some(Latex::Text(process_children(elm, None)?)))
    } else if stag.ends_with("Pr") {
        Ok(Some(Latex::Pr(Pr::new(elm))))
    } else {
        Ok(None)
    }
}

fn call_method(elm: &XmlElement, stag: &str) -> anyhow::Result<Option<Latex>> {
    let text = match stag {
        "acc" => do_acc(elm)?,
        "r" => do_r(elm)?,
        "bar" => do_bar(elm)?,
        "sub" => do_sub(elm)?,
        "sup" => do_sup(elm)?,
        "f" => do_f(elm)?,
        "func" => do_func(elm)?,
        "fName" => do_fname(elm)?,
        "groupChr" => do_groupchr(elm)?,
        "d" => do_d(elm)?,
        "rad" => do_rad(elm)?,
        "eqArr" => do_eqarr(elm)?,
        "limLow" => do_limlow(elm)?,
        "limUpp" => do_limupp(elm)?,
        "lim" => do_lim(elm)?,
        "m" => do_m(elm)?,
        "mr" => do_mr(elm)?,
        "nary" => do_nary(elm)?,
        _ => return Ok(None),
    };
    Ok(Some(Latex::Text(text)))
}

fn require<'a>(c_dict: &'a HashMap<String, Latex>, key: &str) -> anyhow::Result<&'a Latex> {
    c_dict.get(key).ok_or_else(|| anyhow::anyhow!("'{key}'"))
}

/// The accent function.
fn do_acc(elm: &XmlElement) -> anyhow::Result<String> {
    let c_dict = process_children_dict(elm, None)?;
    let pr = require(&c_dict, "accPr")?.pr()?;
    let latex_s = get_val(pr.get("chr"), Some(CHR_DEFAULT_ACC_VAL), CHR).unwrap_or_default();
    py_format(&latex_s, &[require(&c_dict, "e")?.as_str().to_string()], &[])
}

/// The bar function.
fn do_bar(elm: &XmlElement) -> anyhow::Result<String> {
    let c_dict = process_children_dict(elm, None)?;
    let pr = require(&c_dict, "barPr")?.pr()?;
    let latex_s = get_val(pr.get("pos"), Some(POS_DEFAULT_BAR_VAL), POS).unwrap_or_default();
    Ok(format!("{}{}", pr.text, py_format(&latex_s, &[require(&c_dict, "e")?.as_str().to_string()], &[])?))
}

/// The delimiter object.
fn do_d(elm: &XmlElement) -> anyhow::Result<String> {
    let c_dict = process_children_dict(elm, None)?;
    let pr = require(&c_dict, "dPr")?.pr()?;
    let s_val = get_val(pr.get("begChr"), Some(D_DEFAULT_LEFT), T).unwrap_or_default();
    let e_val = get_val(pr.get("endChr"), Some(D_DEFAULT_RIGHT), T).unwrap_or_default();
    let left = if s_val.is_empty() { D_DEFAULT_NULL.to_string() } else { escape_latex(&s_val) };
    let right = if e_val.is_empty() { D_DEFAULT_NULL.to_string() } else { escape_latex(&e_val) };
    let text = require(&c_dict, "e")?.as_str().to_string();
    Ok(format!("{}{}", pr.text, py_format(D, &[], &[("left", left), ("text", text), ("right", right)])?))
}

fn do_sub(elm: &XmlElement) -> anyhow::Result<String> {
    let text = process_children(elm, None)?;
    py_format(SUB, &[text], &[])
}

fn do_sup(elm: &XmlElement) -> anyhow::Result<String> {
    let text = process_children(elm, None)?;
    py_format(SUP, &[text], &[])
}

/// The fraction object.
fn do_f(elm: &XmlElement) -> anyhow::Result<String> {
    let c_dict = process_children_dict(elm, None)?;
    let pr = require(&c_dict, "fPr")?.pr()?;
    let latex_s = get_val(pr.get("type"), Some(F_DEFAULT), F).unwrap_or_default();
    let num = py_none_or(c_dict.get("num").map(Latex::as_str));
    let den = py_none_or(c_dict.get("den").map(Latex::as_str));
    Ok(format!("{}{}", pr.text, py_format(&latex_s, &[], &[("num", num), ("den", den)])?))
}

/// The Function-Apply object (Examples: sin cos).
fn do_func(elm: &XmlElement) -> anyhow::Result<String> {
    let c_dict = process_children_dict(elm, None)?;
    let Some(func_name) = c_dict.get("fName") else {
        anyhow::bail!("'NoneType' object has no attribute 'replace'");
    };
    let Some(e) = c_dict.get("e") else {
        anyhow::bail!("replace() argument 2 must be str, not None");
    };
    Ok(func_name.require_str()?.replace(FUNC_PLACE, e.require_str()?))
}

/// The function name.
fn do_fname(elm: &XmlElement) -> anyhow::Result<String> {
    let mut latex_chars = Vec::new();
    for (stag, t, _) in process_children_list(elm, None)? {
        if stag == "r" {
            match lookup(FUNC, t.as_str()) {
                Some(func) => latex_chars.push(func.to_string()),
                None => anyhow::bail!("Not support func {}", t.as_str()),
            }
        } else {
            latex_chars.push(t.require_str()?.to_string());
        }
    }
    let t = latex_chars.join(BLANK);
    // do_func will replace this
    Ok(if t.contains(FUNC_PLACE) { t } else { format!("{t}{FUNC_PLACE}") })
}

/// The Group-Character object.
fn do_groupchr(elm: &XmlElement) -> anyhow::Result<String> {
    let c_dict = process_children_dict(elm, None)?;
    let pr = require(&c_dict, "groupChrPr")?.pr()?;
    let Some(latex_s) = get_val(pr.get("chr"), None, CHR) else {
        anyhow::bail!("'NoneType' object has no attribute 'format'");
    };
    Ok(format!("{}{}", pr.text, py_format(&latex_s, &[require(&c_dict, "e")?.as_str().to_string()], &[])?))
}

/// The radical object.
fn do_rad(elm: &XmlElement) -> anyhow::Result<String> {
    let c_dict = process_children_dict(elm, None)?;
    let text = py_none_or(c_dict.get("e").map(Latex::as_str));
    match c_dict.get("deg").map(Latex::as_str) {
        Some(deg_text) if !deg_text.is_empty() => {
            py_format(RAD, &[], &[("deg", deg_text.to_string()), ("text", text)])
        }
        _ => py_format(RAD_DEFAULT, &[], &[("text", text)]),
    }
}

/// The Array object.
fn do_eqarr(elm: &XmlElement) -> anyhow::Result<String> {
    let parts = process_children_list(elm, Some(&["e"]))?;
    let mut texts = Vec::with_capacity(parts.len());
    for (_, t, _) in &parts {
        texts.push(t.require_str()?.to_string());
    }
    py_format(ARR, &[], &[("text", texts.join(BRK))])
}

/// The Lower-Limit object.
fn do_limlow(elm: &XmlElement) -> anyhow::Result<String> {
    let t_dict = process_children_dict(elm, Some(&["e", "lim"]))?;
    let e = require(&t_dict, "e")?.as_str();
    match lookup(LIM_FUNC, e) {
        Some(latex_s) => py_format(latex_s, &[], &[("lim", py_none_or(t_dict.get("lim").map(Latex::as_str)))]),
        None => anyhow::bail!("Not support lim {e}"),
    }
}

/// The Upper-Limit object.
fn do_limupp(elm: &XmlElement) -> anyhow::Result<String> {
    let t_dict = process_children_dict(elm, Some(&["e", "lim"]))?;
    py_format(
        LIM_UPP,
        &[],
        &[
            ("lim", py_none_or(t_dict.get("lim").map(Latex::as_str))),
            ("text", py_none_or(t_dict.get("e").map(Latex::as_str))),
        ],
    )
}

/// The lower limit of the limLow object and the upper limit of the limUpp function.
fn do_lim(elm: &XmlElement) -> anyhow::Result<String> {
    Ok(process_children(elm, None)?.replace(LIM_TO.0, LIM_TO.1))
}

/// The Matrix object.
fn do_m(elm: &XmlElement) -> anyhow::Result<String> {
    let mut rows = Vec::new();
    for (stag, t, _) in process_children_list(elm, None)? {
        if stag == "mr" {
            rows.push(t.require_str()?.to_string());
        }
    }
    py_format(M, &[], &[("text", rows.join(BRK))])
}

/// A single row of the matrix m.
fn do_mr(elm: &XmlElement) -> anyhow::Result<String> {
    let parts = process_children_list(elm, Some(&["e"]))?;
    let mut texts = Vec::with_capacity(parts.len());
    for (_, t, _) in &parts {
        texts.push(t.require_str()?.to_string());
    }
    Ok(texts.join(ALN))
}

/// The n-ary object.
fn do_nary(elm: &XmlElement) -> anyhow::Result<String> {
    let mut res = Vec::new();
    let mut bo: Option<String> = Some(String::new());
    for (stag, t, _) in process_children_list(elm, None)? {
        if stag == "naryPr" {
            bo = get_val(t.pr()?.get("chr"), None, CHR_BO);
        } else {
            res.push(t.require_str()?.to_string());
        }
    }
    let Some(bo) = bo else {
        anyhow::bail!("unsupported operand type(s) for +: 'NoneType' and 'str'");
    };
    Ok(format!("{bo}{}", res.join(BLANK)))
}

/// Get text from the 'r' element and try to convert it to LaTeX symbols.
fn do_r(elm: &XmlElement) -> anyhow::Result<String> {
    // ElementTree `findtext("./m:t")`: the text of the first `m:t` child.
    let Some(t) = elm.child("m:t") else {
        anyhow::bail!("'NoneType' object is not iterable");
    };
    let text: String = t
        .children
        .iter()
        .take_while(|c| matches!(c, XmlNode::Text(_)))
        .map(|c| match c {
            XmlNode::Text(s) => s.as_str(),
            XmlNode::Element(_) => "",
        })
        .collect();
    let mut out = String::new();
    for s in text.chars() {
        let key = s.to_string();
        out.push_str(lookup(T, &key).unwrap_or(&key));
    }
    Ok(escape_latex(&out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parsers::markitdown::ooxml::parse_xml;

    const NS: &str = r#"xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math""#;

    fn latex(body: &str) -> anyhow::Result<String> {
        let xml = format!("<m:oMath {NS}>{body}</m:oMath>");
        omath_to_latex(&parse_xml(xml.as_bytes()).unwrap())
    }

    #[test]
    fn fraction_and_scripts() {
        let out = latex(
            "<m:f><m:fPr><m:ctrlPr/></m:fPr><m:num><m:r><m:t>a</m:t></m:r></m:num><m:den><m:r><m:t>b</m:t></m:r></m:den></m:f>\
             <m:sSup><m:e><m:r><m:t>x</m:t></m:r></m:e><m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup>",
        )
        .unwrap();
        assert_eq!(out, "\\frac{a}{b}x^{2}");
    }

    #[test]
    fn missing_fraction_properties_raise() {
        assert!(latex("<m:f><m:num><m:r><m:t>a</m:t></m:r></m:num></m:f>").is_err());
    }

    #[test]
    fn python_format_semantics() {
        assert_eq!(py_format("\\hat{{{0}}}", &["x".into()], &[]).unwrap(), "\\hat{x}");
        assert!(py_format("\\ocirc{{{0}}}}", &["x".into()], &[]).is_err());
        assert_eq!(py_format("→", &["x".into()], &[]).unwrap(), "→");
        assert_eq!(escape_latex("a_b\\_c{"), "a\\_b\\_c\\{");
    }

    #[test]
    fn delimiter_and_nary() {
        let out = latex(
            "<m:d><m:dPr><m:begChr m:val=\"[\"/><m:endChr m:val=\"\"/></m:dPr><m:e><m:r><m:t>x</m:t></m:r></m:e></m:d>\
             <m:nary><m:naryPr><m:chr m:val=\"\u{2211}\"/></m:naryPr><m:sub><m:r><m:t>i</m:t></m:r></m:sub><m:e><m:r><m:t>i</m:t></m:r></m:e></m:nary>",
        )
        .unwrap();
        assert_eq!(out, "\\left[x\\right.\\sum_{i}i");
    }
}
