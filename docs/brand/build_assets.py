"""Build the canonical, path-only Algua brand assets.

Run from the repository root:
    uv run python docs/brand/build_assets.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

OBSIDIAN = "#000000"
ELECTRIC = "#1267FF"
ELECTRIC_DARK = "#3982FF"
ICE = "#F7F9FC"
FOG = "#A9B4C5"

# All terminals follow the A's 116:384 outer-leg slope. The recessed sea
# horizon is behind the A; the foreground horizon is in front.
RECESSED_SEA_PATH = "M190.5 299H321.5L323.6 306H188.4Z"
A_AND_WATER_PATH = (
    "M256 48L372 432H344L256 140.7L168 432H140Z"
    "M188.4 306H323.6L337.7 312H174.3Z"
)
FOREGROUND_SEA_PATH = "M176.3 312H335.7L338.4 321H173.6Z"

# Fira Sans ExtraLight, optically tracked for Algua and converted to one
# compound outline. No installed font is needed to render the assets.
WORDMARK_PATH = (
    "M491 0H542L311 684H251L20 0H68L131 192H428Z"
    "M145 234 281 645 415 234Z"
    "M768-11C788-11 808-5 823 2L810 37C799 33 786 30 773 30"
    "C743 30 728 48 728 84V741L682 735V82C682 19 717-11 768-11Z"
    "M1344 562C1297 543 1263 531 1119 532C1005 532 932 462 932 358"
    "C932 287 960 241 1019 210C984 188 962 158 962 123C962 73 1005 33 1089 33"
    "H1174C1251 33 1297 0 1297-58C1297-123 1248-163 1124-163"
    "C995-163 954-134 953-56H909C909-159 971-203 1124-203"
    "C1273-203 1344-147 1344-55C1344 21 1283 74 1181 74H1095"
    "C1029 74 1007 99 1007 132C1007 158 1024 181 1051 196"
    "C1071 189 1095 185 1120 185C1235 185 1305 254 1305 355"
    "C1305 431 1270 476 1217 501C1282 502 1326 506 1359 516Z"
    "M1120 495C1213 495 1258 440 1258 356C1258 270 1208 221 1119 221"
    "C1036 221 979 269 979 358C979 438 1024 495 1120 495Z"
    "M1853 522H1807V141C1770 76 1720 29 1647 29C1575 29 1536 66 1536 153"
    "V522H1490V149C1490 46 1544-11 1638-11C1721-11 1776 35 1811 97L1814 0H1853Z"
    "M2403 113V370C2403 475 2352 532 2236 532C2183 532 2133 520 2077 498"
    "L2090 461C2141 481 2188 492 2232 492C2323 492 2357 452 2357 367V312H2250"
    "C2121 312 2038 253 2038 145C2038 52 2098-11 2198-11C2270-11 2325 20 2363 81"
    "C2368 24 2393 0 2436-11L2447 22C2418 33 2403 51 2403 113Z"
    "M2204 28C2131 28 2087 71 2087 146C2087 234 2150 276 2257 276H2357V127"
    "C2324 64 2275 28 2204 28Z"
)

# Fira Sans Medium product descriptor, tracked and converted to outlines.
DESCRIPTOR_PATH = (
    "M447 0H587L373 691H210L-5 0H132L175 160H404Z"
    "M201 260 289 590 377 260Z"
    "M1033 706C860 706 716 580 716 346C716 109 824-15 1021-15C1108-15 1190 11 1259 53"
    "V391H1011L1025 293H1129V113C1096 94 1058 86 1019 86C911 86 855 157 855 346"
    "C855 532 939 606 1037 606C1096 606 1135 588 1182 551L1253 625C1195 674 1129 706 1033 706Z"
    "M1883 691H1482V0H1887V99H1614V302H1836V400H1614V593H1869Z"
    "M2612 691H2492V367C2492 269 2504 169 2510 131L2269 691H2102V0H2222V282"
    "C2222 410 2211 498 2204 559L2441 0H2612Z"
    "M3308 691H2795V586H2981V0H3114V586H3294Z"
    "M3455 262H3757V364H3455Z"
    "M3979 0H4111V288H4324V386H4111V593H4356L4370 691H3979Z"
    "M4690 691H4558V0H4690Z"
    "M5311 0H5463L5282 301C5376 338 5422 394 5422 488"
    "C5422 625 5329 691 5148 691H4942V0H5074V277H5158Z"
    "M5074 595H5145C5241 595 5284 563 5284 488C5284 405 5238 372 5156 372H5074Z"
    "M5842 706C5706 706 5607 630 5607 515C5607 410 5670 352 5815 307C5923 274 5952 248 5952 190"
    "C5952 123 5899 87 5825 87C5753 87 5698 113 5647 156L5579 80C5636 24 5718-15 5828-15"
    "C5993-15 6090 73 6090 197C6090 327 6006 377 5885 415C5770 451 5743 473 5743 523"
    "C5743 577 5788 605 5850 605C5907 605 5955 587 6005 546L6070 621C6010 676 5943 706 5842 706Z"
    "M6737 691H6224V586H6410V0H6543V586H6723Z"
    "M7619 0H7759L7545 691H7382L7167 0H7304L7347 160H7576Z"
    "M7373 260 7461 590 7549 260Z"
    "M8058 691H7926V0H8320L8335 108H8058Z"
    "M8801 706C8628 706 8484 580 8484 346C8484 109 8592-15 8789-15C8876-15 8958 11 9027 53"
    "V391H8779L8793 293H8897V113C8864 94 8826 86 8787 86C8679 86 8623 157 8623 346"
    "C8623 532 8707 606 8805 606C8864 606 8903 588 8950 551L9021 625C8963 674 8897 706 8801 706Z"
    "M9515 706C9330 706 9212 572 9212 345C9212 114 9330-15 9515-15C9700-15 9817 118 9817 346"
    "C9817 577 9700 706 9515 706ZM9515 605C9619 605 9678 531 9678 346C9678 159 9618 86 9515 86"
    "C9414 86 9351 159 9351 345C9351 531 9412 605 9515 605Z"
    "M10400 0H10552L10371 301C10465 338 10511 394 10511 488"
    "C10511 625 10418 691 10237 691H10031V0H10163V277H10247Z"
    "M10163 595H10234C10330 595 10373 563 10373 488C10373 405 10327 372 10245 372H10163Z"
    "M10864 691H10732V0H10864Z"
    "M11560 691H11047V586H11233V0H11366V586H11546Z"
    "M12114 0H12247V691H12114V414H11868V691H11736V0H11868V309H12114Z"
    "M13099 691H12927L12811 207L12688 691H12518L12463 0H12591L12608 285"
    "C12614 384 12616 479 12612 572L12746 75H12871L12997 571"
    "C12995 498 13000 391 13006 289L13024 0H13153Z"
    "M13502 691H13370V0H13502Z"
    "M14018 706C13848 706 13716 579 13716 347C13716 112 13841-15 14020-15"
    "C14114-15 14186 26 14228 69L14166 147C14126 117 14085 91 14025 91"
    "C13927 91 13855 165 13855 347C13855 534 13929 604 14025 604"
    "C14072 604 14113 587 14153 555L14220 633C14163 680 14106 706 14018 706Z"
    "M15121 0H15273L15092 301C15186 338 15232 394 15232 488"
    "C15232 625 15139 691 14958 691H14752V0H14884V277H14968Z"
    "M14884 595H14955C15051 595 15094 563 15094 488C15094 405 15048 372 14966 372H14884Z"
    "M15854 691H15453V0H15858V99H15585V302H15807V400H15585V593H15840Z"
    "M16272 706C16136 706 16037 630 16037 515C16037 410 16100 352 16245 307"
    "C16353 274 16382 248 16382 190"
    "C16382 123 16329 87 16255 87C16183 87 16128 113 16077 156L16009 80C16066 24 16148-15 16258-15"
    "C16423-15 16520 73 16520 197C16520 327 16436 377 16315 415C16200 451 16173 473 16173 523"
    "C16173 577 16218 605 16280 605C16337 605 16385 587 16435 546"
    "L16500 621C16440 676 16373 706 16272 706Z"
    "M17124 691H16723V0H17128V99H16855V302H17077V400H16855V593H17110Z"
    "M17709 0H17849L17635 691H17472L17257 0H17394L17437 160H17666Z"
    "M17463 260 17551 590 17639 260Z"
    "M18385 0H18537L18356 301C18450 338 18496 394 18496 488"
    "C18496 625 18403 691 18222 691H18016V0H18148V277H18232Z"
    "M18148 595H18219C18315 595 18358 563 18358 488C18358 405 18312 372 18230 372H18148Z"
    "M18981 706C18811 706 18679 579 18679 347C18679 112 18804-15 18983-15"
    "C19077-15 19149 26 19191 69L19129 147C19089 117 19048 91 18988 91"
    "C18890 91 18818 165 18818 347C18818 534 18892 604 18988 604"
    "C19035 604 19076 587 19116 555L19183 633C19126 680 19069 706 18981 706Z"
    "M19753 0H19886V691H19753V414H19507V691H19375V0H19507V309H19753Z"
)


def svg_shell(view_box: str, title: str, desc: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" '
        'role="img" aria-labelledby="title desc">\n'
        f'  <title id="title">{title}</title>\n'
        f'  <desc id="desc">{desc}</desc>\n'
        f"{body}\n"
        "</svg>\n"
    )


def mark_paths(ink: str, blue: str, indent: str = "  ") -> str:
    return "\n".join(
        (
            f'{indent}<path d="{RECESSED_SEA_PATH}" fill="{blue}"/>',
            f'{indent}<path d="{A_AND_WATER_PATH}" fill="{ink}"/>',
            f'{indent}<path d="{FOREGROUND_SEA_PATH}" fill="{blue}"/>',
        )
    )


def write(name: str, content: str) -> None:
    (ROOT / name).write_text(content, encoding="utf-8")


def build() -> None:
    write(
        "logo-mark.svg",
        svg_shell(
            "0 0 512 512",
            "Algua logo mark",
            "A precise capital A emerging from a layered electric-blue sea plane.",
            mark_paths(OBSIDIAN, ELECTRIC),
        ),
    )
    write(
        "logo-mark-reversed.svg",
        svg_shell(
            "0 0 512 512",
            "Algua reversed logo mark",
            "A precise white capital A emerging from a layered electric-blue sea plane.",
            mark_paths(ICE, ELECTRIC_DARK),
        ),
    )

    horizontal_light = "\n".join(
        (
            '  <g transform="translate(60 28) scale(.59)">',
            mark_paths(OBSIDIAN, ELECTRIC, "    "),
            "  </g>",
            f'  <path d="{WORDMARK_PATH}" fill="{OBSIDIAN}" '
            'transform="translate(340 235) scale(.245 -.245)"/>',
        )
    )
    write(
        "logo-horizontal.svg",
        svg_shell(
            "0 0 1100 360",
            "Algua horizontal logo",
            "Outlined Algua wordmark beside the Sea A symbol.",
            horizontal_light,
        ),
    )

    horizontal_dark = "\n".join(
        (
            '  <g transform="translate(60 28) scale(.59)">',
            mark_paths(ICE, ELECTRIC_DARK, "    "),
            "  </g>",
            f'  <path d="{WORDMARK_PATH}" fill="{ICE}" '
            'transform="translate(340 235) scale(.245 -.245)"/>',
        )
    )
    write(
        "logo-horizontal-reversed.svg",
        svg_shell(
            "0 0 1100 360",
            "Algua reversed horizontal logo",
            "Outlined Algua wordmark beside the Sea A symbol for dark backgrounds.",
            horizontal_dark,
        ),
    )

    avatar_body = "\n".join(
        (
            f'  <rect width="1024" height="1024" rx="224" fill="{OBSIDIAN}"/>',
            '  <g transform="translate(115.2 140) scale(1.55)">',
            mark_paths(ICE, ELECTRIC_DARK, "    "),
            "  </g>",
        )
    )
    write(
        "social-avatar.svg",
        svg_shell(
            "0 0 1024 1024",
            "Algua social avatar",
            "The Algua Sea A centered on an Obsidian square.",
            avatar_body,
        ),
    )

    favicon_body = "\n".join(
        (
            f'  <rect width="512" height="512" rx="112" fill="{OBSIDIAN}"/>',
            mark_paths(ICE, ELECTRIC_DARK),
        )
    )
    write(
        "favicon.svg",
        svg_shell(
            "0 0 512 512",
            "Algua favicon",
            "The exact Algua Sea A on an Obsidian tile.",
            favicon_body,
        ),
    )

    banner_body = "\n".join(
        (
            "  <defs>",
            '    <linearGradient id="glow" x1="0" x2="1">',
            f'      <stop offset="0" stop-color="{ELECTRIC}" stop-opacity="0"/>',
            f'      <stop offset=".5" stop-color="{ELECTRIC}" stop-opacity=".28"/>',
            f'      <stop offset="1" stop-color="{ELECTRIC}" stop-opacity="0"/>',
            "    </linearGradient>",
            "  </defs>",
            f'  <rect width="1600" height="480" rx="28" fill="{OBSIDIAN}"/>',
            '  <path d="M0 367H1600V369H0Z" fill="url(#glow)"/>',
            '  <g transform="translate(90 55) scale(.66)">',
            mark_paths(ICE, ELECTRIC_DARK, "    "),
            "  </g>",
            f'  <path d="{WORDMARK_PATH}" fill="{ICE}" '
            'transform="translate(400 245) scale(.25 -.25)"/>',
            f'  <path d="{DESCRIPTOR_PATH}" fill="{FOG}" '
            'transform="translate(407 329) scale(.03 -.03)"/>',
        )
    )
    write(
        "readme-banner.svg",
        svg_shell(
            "0 0 1600 480",
            "Algua brand banner",
            "The Algua Sea A, outlined wordmark, and product descriptor on Obsidian.",
            banner_body,
        ),
    )


if __name__ == "__main__":
    build()
