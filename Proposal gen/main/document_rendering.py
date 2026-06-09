"""DOCX rendering, style, chart, and logo helpers."""
from __future__ import annotations

from .proposal_shared import *
from .research import Researcher
from .text_hygiene import clean_markup_artifacts
from .reader_facing_hygiene import sanitize_reader_facing_sources

class LogoManager:
    BLOCKED_LOGO_HOSTS = {
        "wikipedia.org", "wikimedia.org", "facebook.com", "instagram.com", "linkedin.com",
        "youtube.com", "companieshouse.id", "ipaddress.com", "reddit.com",
    }

    @staticmethod
    def _create_fallback_logo(client_name: str) -> io.BytesIO:
        initials = "".join([w[0] for w in re.findall(r"[A-Za-z0-9]+", client_name)[:3]]).upper() or "CL"
        canvas = Image.new('RGBA', (420, 180), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 88)
        except Exception:
            font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), initials, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = draw.textsize(initials, font=font)
        draw.text(((420 - w) / 2, (180 - h) / 2 - 4), initials, fill=(15, 23, 42, 255), font=font)
        out = io.BytesIO()
        canvas.save(out, format='PNG')
        out.seek(0)
        return out

    @staticmethod
    def _crop_transparent_bounds(img: Image.Image, padding: int = 4) -> Image.Image:
        alpha = img.getchannel("A")
        bbox = alpha.getbbox()
        if not bbox:
            return img
        left, top, right, bottom = bbox
        return img.crop((
            max(0, left - padding),
            max(0, top - padding),
            min(img.width, right + padding),
            min(img.height, bottom + padding),
        ))

    @staticmethod
    def _normalize_logo_image(img: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int]]:
        rgba = img.convert("RGBA")
        alpha_min, alpha_max = rgba.getchannel("A").getextrema()

        if alpha_min < 250:
            normalized = LogoManager._crop_transparent_bounds(rgba)
        else:
            width, height = rgba.size
            border_pixels: List[Tuple[int, int, int]] = []
            step_x = max(1, width // 24)
            step_y = max(1, height // 24)
            for x in range(0, width, step_x):
                border_pixels.append(rgba.getpixel((x, 0))[:3])
                border_pixels.append(rgba.getpixel((x, height - 1))[:3])
            for y in range(0, height, step_y):
                border_pixels.append(rgba.getpixel((0, y))[:3])
                border_pixels.append(rgba.getpixel((width - 1, y))[:3])

            if border_pixels:
                background = tuple(
                    int(sum(pixel[idx] for pixel in border_pixels) / len(border_pixels))
                    for idx in range(3)
                )
                max_border_delta = max(
                    max(abs(pixel[idx] - background[idx]) for idx in range(3))
                    for pixel in border_pixels
                )
            else:
                background = (255, 255, 255)
                max_border_delta = 255

            normalized = rgba
            if max_border_delta <= 28:
                tolerance = 34
                transparent_data = []
                for r, g, b, a in rgba.getdata():
                    if a <= 8:
                        transparent_data.append((r, g, b, 0))
                        continue
                    if max(abs(r - background[0]), abs(g - background[1]), abs(b - background[2])) <= tolerance:
                        transparent_data.append((r, g, b, 0))
                    else:
                        transparent_data.append((r, g, b, a))
                candidate = Image.new("RGBA", rgba.size)
                candidate.putdata(transparent_data)
                cropped = LogoManager._crop_transparent_bounds(candidate)
                if cropped.getchannel("A").getbbox():
                    normalized = cropped

        visible_pixels = [pixel[:3] for pixel in normalized.getdata() if pixel[3] > 32]
        if visible_pixels:
            dominant = tuple(
                int(sum(pixel[idx] for pixel in visible_pixels) / len(visible_pixels))
                for idx in range(3)
            )
        else:
            dominant = tuple(int(value) for value in ImageStat.Stat(normalized.convert("RGB")).mean[:3])

        return normalized, dominant

    @staticmethod
    def _border_opacity_ratio(img: Image.Image) -> float:
        rgba = img.convert("RGBA")
        width, height = rgba.size
        if width <= 1 or height <= 1:
            return 1.0

        border_coords = set()
        for x in range(width):
            border_coords.add((x, 0))
            border_coords.add((x, height - 1))
        for y in range(height):
            border_coords.add((0, y))
            border_coords.add((width - 1, y))

        opaque_count = 0
        for x, y in border_coords:
            if rgba.getpixel((x, y))[3] > 32:
                opaque_count += 1
        return opaque_count / max(1, len(border_coords))

    @staticmethod
    def _logo_identity_terms(client_name: str) -> List[str]:
        ignored = {
            "pt", "cv", "tbk", "persero", "company", "corp", "corporate",
            "indonesia", "indo", "the", "and", "bank", "dinas", "kabupaten",
            "kota", "provinsi", "pemerintah",
        }
        legal_ignored = {"pt", "cv", "tbk", "persero", "company", "corp", "corporate", "the", "and"}
        acronym_words = [
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]+", client_name or "")
            if len(word) >= 3 and word.lower() not in legal_ignored
        ]
        words = [
            word.lower()
            for word in re.findall(r"[A-Za-z0-9]+", client_name or "")
            if len(word) >= 3 and word.lower() not in ignored
        ]
        terms: List[str] = []
        seen = set()
        for word in words:
            if word not in seen:
                seen.add(word)
                terms.append(word)
        if len(acronym_words) >= 2:
            acronym = "".join(word[0] for word in acronym_words[:4])
            if len(acronym) >= 3 and acronym not in seen:
                terms.append(acronym)
        return terms

    @staticmethod
    def _logo_metadata_text(item: Dict[str, Any]) -> str:
        values = [
            item.get("title", ""),
            item.get("source", ""),
            item.get("link", ""),
            item.get("imageUrl", ""),
        ]
        return " ".join(re.sub(r"[^A-Za-z0-9]+", " ", str(value or "")).lower() for value in values)

    @staticmethod
    def _logo_identity_score(item: Dict[str, Any], client_name: str) -> float:
        terms = LogoManager._logo_identity_terms(client_name)
        if not terms:
            return 0.0
        metadata = LogoManager._logo_metadata_text(item)
        source_host = ""
        for key in ("source", "link", "imageUrl"):
            try:
                source_host += " " + (urlparse(str(item.get(key) or "")).netloc or "")
            except Exception:
                continue
        source_host = re.sub(r"[^A-Za-z0-9]+", " ", source_host).lower()
        hit_count = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", metadata))
        host_hit_count = sum(1 for term in terms if term in source_host)
        required_hits = 1 if len(terms) <= 1 else 2
        if (hit_count + host_hit_count) < required_hits:
            return 0.0
        score = float(hit_count) + (0.75 * host_hit_count)
        if "logo" in metadata:
            score += 0.35
        return score

    @staticmethod
    def _host_allowed_for_logo(host: str) -> bool:
        raw_host = (host or "").lower().strip()
        if "://" in raw_host:
            raw_host = urlparse(raw_host).netloc
        normalized = raw_host.split("@")[-1].split(":")[0].strip(".")
        if normalized.startswith("www."):
            normalized = normalized[4:]
        if not normalized:
            return False
        return not any(
            normalized == blocked or normalized.endswith(f".{blocked}")
            for blocked in LogoManager.BLOCKED_LOGO_HOSTS
        )

    @staticmethod
    def _logo_search_item_allowed(item: Dict[str, Any]) -> bool:
        for key in ("link", "imageUrl"):
            value = str(item.get(key) or "").strip()
            if not value:
                continue
            host = urlparse(value).netloc
            if host and not LogoManager._host_allowed_for_logo(host):
                return False
        source = str(item.get("source") or "").strip()
        if "." in source and not LogoManager._host_allowed_for_logo(source):
            return False
        return True

    @staticmethod
    def _official_domain_candidates(client_name: str) -> List[str]:
        candidates: List[str] = []
        seen = set()

        def add(host: str) -> None:
            clean = re.sub(r"^www\.", "", (host or "").lower().strip())
            if not LogoManager._host_allowed_for_logo(clean) or clean in seen:
                return
            seen.add(clean)
            candidates.append(clean)

        if Researcher._has_serper_key():
            try:
                payload = json.dumps({"q": f"{client_name} official website", "num": 8})
                headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
                res = requests.post("https://google.serper.dev/search", headers=headers, data=payload, timeout=8).json()
                for item in res.get("organic", []) or []:
                    link = str(item.get("link") or "")
                    title = str(item.get("title") or "")
                    snippet = str(item.get("snippet") or "")
                    host = urlparse(link).netloc
                    haystack = f"{title} {snippet} {host}".lower()
                    if "official" in haystack or any(term in haystack for term in LogoManager._logo_identity_terms(client_name)):
                        add(host)
            except Exception as exc:
                logger.debug("Official logo domain discovery failed: %s", exc)
        return candidates[:4]

    @staticmethod
    def _image_stream_from_url(image_url: str, client_name: str, identity_boost: float = 1.0) -> Optional[Tuple[float, io.BytesIO, Tuple[int, int, int]]]:
        try:
            img_resp = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=6)
            if img_resp.status_code != 200 or not img_resp.content:
                return None
            stream = io.BytesIO(img_resp.content)
            img = Image.open(stream)
            img, dom_color = LogoManager._normalize_logo_image(img)
            if img.width < 24 or img.height < 24:
                return None
            img.thumbnail((640, 260))
            border_ratio = LogoManager._border_opacity_ratio(img)
            if border_ratio > 0.28:
                return None
            luminance = 0.299 * dom_color[0] + 0.587 * dom_color[1] + 0.114 * dom_color[2]
            if luminance > 120:
                factor = 120 / luminance
                dom_color = tuple(max(0, min(255, int(c * factor))) for c in dom_color)
            png_stream = io.BytesIO()
            img.save(png_stream, format='PNG')
            png_stream.seek(0)
            aspect_penalty = 0.01 * abs((img.width / max(1, img.height)) - 3.0)
            size_bonus = min(0.18, (img.width * img.height) / 800000.0)
            score = border_ratio + aspect_penalty - (0.16 * identity_boost) - size_bonus
            return score, png_stream, tuple(dom_color)
        except Exception:
            return None

    @staticmethod
    def _fetch_logo_from_domain(domain: str, client_name: str) -> Optional[Tuple[float, io.BytesIO, Tuple[int, int, int]]]:
        urls: List[str] = []
        for base in [f"https://{domain}", f"https://www.{domain}"]:
            try:
                resp = requests.get(base, headers={'User-Agent': 'Mozilla/5.0'}, timeout=6)
                if resp.status_code >= 400 or not resp.text:
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                for img in soup.find_all("img"):
                    metadata = " ".join(str(img.get(attr) or "") for attr in ("src", "alt", "class", "id", "title")).lower()
                    if "logo" in metadata or any(term in metadata for term in LogoManager._logo_identity_terms(client_name)):
                        src = str(img.get("src") or "").strip()
                        if src:
                            urls.append(urljoin(base, src))
                for tag in soup.find_all("link"):
                    rel = " ".join(tag.get("rel") or []).lower()
                    href = str(tag.get("href") or "").strip()
                    if href and any(token in rel for token in ("icon", "apple-touch-icon")):
                        urls.append(urljoin(base, href))
                for tag in soup.find_all("meta"):
                    prop = str(tag.get("property") or tag.get("name") or "").lower()
                    content = str(tag.get("content") or "").strip()
                    if content and prop in {"og:image", "twitter:image"}:
                        urls.append(urljoin(base, content))
                break
            except Exception:
                continue
        urls.extend([
            f"https://logo.clearbit.com/{domain}",
            f"https://www.google.com/s2/favicons?sz=256&domain={domain}",
        ])
        best: Optional[Tuple[float, io.BytesIO, Tuple[int, int, int]]] = None
        seen = set()
        for image_url in urls:
            if image_url in seen:
                continue
            seen.add(image_url)
            candidate = LogoManager._image_stream_from_url(image_url, client_name, identity_boost=1.4)
            if candidate and (best is None or candidate[0] < best[0]):
                best = candidate
        return best

    @staticmethod
    def get_logo_and_color(client_name: str) -> Tuple[Optional[io.BytesIO], Tuple[int, int, int]]:
        try:
            official_domains = LogoManager._official_domain_candidates(client_name)
        except Exception as exc:
            logger.debug("Official logo domain discovery failed: %s", exc)
            official_domains = []
        for domain in official_domains:
            try:
                candidate = LogoManager._fetch_logo_from_domain(domain, client_name)
            except Exception as exc:
                logger.debug("Official logo fetch failed for %s: %s", domain, exc)
                continue
            if candidate is not None:
                return candidate[1], candidate[2]
        if not Researcher._has_serper_key():
            return LogoManager._create_fallback_logo(client_name), DEFAULT_COLOR
        try:
            url = "https://google.serper.dev/images"
            payload = json.dumps({"q": f"official {client_name} logo png transparent", "num": 6})
            headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
            res = requests.post(url, headers=headers, data=payload, timeout=8).json()
            
            if 'images' in res and res['images']:
                best_candidate: Optional[Tuple[float, io.BytesIO, Tuple[int, int, int]]] = None
                for item in res['images']:
                    try:
                        if not LogoManager._logo_search_item_allowed(item):
                            continue
                        identity_score = LogoManager._logo_identity_score(item, client_name)
                        if identity_score <= 0:
                            continue
                        img_resp = requests.get(item['imageUrl'], headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                        if img_resp.status_code == 200:
                            stream = io.BytesIO(img_resp.content)
                            img = Image.open(stream)
                            img, dom_color = LogoManager._normalize_logo_image(img)
                            img.thumbnail((600, 240))
                            border_ratio = LogoManager._border_opacity_ratio(img)
                            if border_ratio > 0.18:
                                continue

                            luminance = 0.299 * dom_color[0] + 0.587 * dom_color[1] + 0.114 * dom_color[2]
                            if luminance > 120:
                                factor = 120 / luminance
                                dom_color = [max(0, min(255, int(c * factor))) for c in dom_color]

                            png_stream = io.BytesIO()
                            img.save(png_stream, format='PNG')
                            png_stream.seek(0)
                            score = border_ratio + (0.01 * abs((img.width / max(1, img.height)) - 3.0)) - (0.08 * identity_score)
                            candidate = (score, png_stream, tuple(dom_color))
                            if best_candidate is None or candidate[0] < best_candidate[0]:
                                best_candidate = candidate
                    except Exception:
                        continue
                if best_candidate is not None:
                    return best_candidate[1], best_candidate[2]
        except Exception as e:
            logger.warning(f"Logo Retrieval Error: {e}")
        return LogoManager._create_fallback_logo(client_name), DEFAULT_COLOR


class StyleEngine:
    PROFESSIONAL_FONT = "Times New Roman"
    BODY_FONT_SIZE = 12
    BODY_LINE_SPACING = 1.25
    BODY_SPACE_AFTER = 8
    HEADING_1_SIZE = 14
    HEADING_2_SIZE = 12
    HEADING_3_SIZE = 11
    TABLE_FONT_SIZE = 10.5
    TEXT_COLOR = (0, 0, 0)
    SUBTLE_TEXT_COLOR = (89, 89, 89)
    TABLE_HEADER_FILL = "E7E6E6"

    @staticmethod
    def apply_document_styles(doc: Document, preserve_existing: bool = False) -> None:
        style = doc.styles['Normal']
        if not preserve_existing:
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.BODY_FONT_SIZE)
            pf = style.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = StyleEngine.BODY_LINE_SPACING
            pf.space_after = Pt(StyleEngine.BODY_SPACE_AFTER)
            pf.space_before = Pt(0)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            StyleEngine._apply_enhanced_heading_styles(doc, StyleEngine.TEXT_COLOR)
            StyleEngine._apply_enhanced_list_styles(doc)
            StyleEngine._apply_enhanced_table_styles(doc, StyleEngine.TEXT_COLOR)
        for section in doc.sections:
            if preserve_existing:
                continue
            section.top_margin = Cm(2.54)
            section.bottom_margin = Cm(2.54)
            section.left_margin = Cm(2.54)
            section.right_margin = Cm(2.54)
    
    @staticmethod
    def apply_enhanced_styles(doc: Document, theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        StyleEngine._apply_base_enhanced_styles(doc)
        StyleEngine._apply_enhanced_heading_styles(doc, theme_color)
        StyleEngine._apply_enhanced_list_styles(doc)
        StyleEngine._apply_enhanced_table_styles(doc, theme_color)
    
    @staticmethod
    def _apply_base_enhanced_styles(doc: Document) -> None:
        try:
            style = doc.styles['Normal']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.BODY_FONT_SIZE)
            
            pf = style.paragraph_format
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = StyleEngine.BODY_LINE_SPACING
            pf.space_after = Pt(StyleEngine.BODY_SPACE_AFTER)
            pf.space_before = Pt(0)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        except Exception as e:
            logger.warning(f"Could not apply base styles: {e}")
        
        for section in doc.sections:
            section.top_margin = Cm(2.54)
            section.bottom_margin = Cm(2.54)
            section.left_margin = Cm(2.54)
            section.right_margin = Cm(2.54)
    
    @staticmethod
    def _apply_enhanced_heading_styles(doc: Document, theme_color: Tuple[int, int, int]) -> None:
        try:
            style = doc.styles['Heading 1']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.HEADING_1_SIZE)
            style.font.bold = True
            style.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
            
            pf = style.paragraph_format
            pf.space_before = Pt(14)
            pf.space_after = Pt(8)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.05
            pf.keep_with_next = True
        except Exception as e:
            logger.warning(f"Could not apply Heading 1 style: {e}")
        
        try:
            style = doc.styles['Heading 2']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.HEADING_2_SIZE)
            style.font.bold = True
            style.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
            
            pf = style.paragraph_format
            pf.space_before = Pt(12)
            pf.space_after = Pt(6)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.05
            pf.keep_with_next = True
        except Exception as e:
            logger.warning(f"Could not apply Heading 2 style: {e}")
        
        try:
            style = doc.styles['Heading 3']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.HEADING_3_SIZE)
            style.font.bold = True
            style.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
            
            pf = style.paragraph_format
            pf.space_before = Pt(10)
            pf.space_after = Pt(5)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.05
            pf.keep_with_next = True
        except Exception as e:
            logger.warning(f"Could not apply Heading 3 style: {e}")
    
    @staticmethod
    def _apply_enhanced_list_styles(doc: Document) -> None:
        try:
            style = doc.styles['List Bullet']
            pf = style.paragraph_format
            pf.space_after = Pt(5)
            pf.space_before = Pt(0)
            pf.left_indent = Inches(0.36)
            pf.first_line_indent = Inches(-0.18)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.15
        except Exception as e:
            logger.warning(f"Could not apply List Bullet style: {e}")
        
        try:
            style = doc.styles['List Number']
            pf = style.paragraph_format
            pf.space_after = Pt(5)
            pf.space_before = Pt(0)
            pf.left_indent = Inches(0.36)
            pf.first_line_indent = Inches(-0.18)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.15
        except Exception as e:
            logger.warning(f"Could not apply List Number style: {e}")
    
    @staticmethod
    def _apply_enhanced_table_styles(doc: Document, theme_color: Tuple[int, int, int]) -> None:
        try:
            style = doc.styles['Table Grid']
            style.font.name = StyleEngine.PROFESSIONAL_FONT
            style.font.size = Pt(StyleEngine.TABLE_FONT_SIZE)
        except Exception as e:
            logger.warning(f"Could not apply Table Grid style: {e}")
    
    @staticmethod
    def add_colored_heading(doc: Document, text: str, level: int = 1, 
                           theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        heading = doc.add_heading(text, level=level)
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        for run in heading.runs:
            run.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
            run.font.name = StyleEngine.PROFESSIONAL_FONT
            if level == 1:
                run.font.size = Pt(StyleEngine.HEADING_1_SIZE)
                run.font.bold = True
            elif level == 2:
                run.font.size = Pt(StyleEngine.HEADING_2_SIZE)
                run.font.bold = True
            else:
                run.font.size = Pt(StyleEngine.HEADING_3_SIZE)
                run.font.bold = True
    
    @staticmethod
    def add_horizontal_line(doc: Document, color: Tuple[int, int, int] = (200, 200, 200)) -> None:
        try:
            paragraph = doc.add_paragraph()
            pPr = paragraph._element.get_or_add_pPr()
            pBdr = OxmlElement('w:pBdr')
            
            bottom = OxmlElement('w:bottom')
            bottom.set(qn('w:val'), 'single')
            bottom.set(qn('w:sz'), '12')
            bottom.set(qn('w:space'), '1')
            bottom.set(qn('w:color'), '%02x%02x%02x' % color)
            
            pBdr.append(bottom)
            pPr.append(pBdr)
            
            paragraph.paragraph_format.space_after = Pt(6)
            paragraph.paragraph_format.space_before = Pt(6)
        except Exception as e:
            logger.warning(f"Could not add horizontal line: {e}")
    
    @staticmethod
    def add_info_box(doc: Document, title: str, content: str, 
                    theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.font.bold = True
        run.font.size = Pt(11)
        run.font.name = StyleEngine.PROFESSIONAL_FONT
        run.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
        
        p = doc.add_paragraph(content)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(6)
    
    @staticmethod
    def format_contact_block(doc: Document, contact_lines: list, 
                           theme_color: Tuple[int, int, int] = (30, 58, 138)) -> None:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(4)
        
        run = p.add_run("Kontak Resmi")
        run.font.bold = True
        run.font.size = Pt(11)
        run.font.name = StyleEngine.PROFESSIONAL_FONT
        run.font.color.rgb = RGBColor(*StyleEngine.TEXT_COLOR)
        
        for line in contact_lines:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(line)
            run.font.name = StyleEngine.PROFESSIONAL_FONT

class ChartEngine:
    @staticmethod
    def _to_matplotlib_rgb(theme_color: Tuple[int, int, int]) -> Tuple[float, float, float]:
        return tuple(c/255 for c in theme_color)

    @staticmethod
    def create_gantt_chart(data_str: str, theme_color: Tuple[int, int, int]) -> Optional[io.BytesIO]:
        try:
            parts = data_str.split('|')
            if len(parts) == 3:
                title_str, unit_str, raw_data = parts[0].strip(), parts[1].strip(), parts[2].strip()
            else:
                title_str, unit_str, raw_data = "Timeline", "Waktu", data_str
            tasks = []
            for p in raw_data.split(';'):
                t_parts = p.split(',')
                if len(t_parts) >= 3:
                    tasks.append({"task": t_parts[0].strip(), "start": float(re.sub(r'[^\d.]', '', t_parts[1])), "dur": float(re.sub(r'[^\d.]', '', t_parts[2]))})
            if not tasks: return None
            tasks = tasks[::-1] 
            
            fig, ax = plt.subplots(figsize=(8.5, max(4, len(tasks)*0.8)))
            for i, task in enumerate(tasks):
                rect = patches.FancyBboxPatch((task['start'], i-0.3), task['dur'], 0.6, boxstyle="round,pad=0.02", ec="#ffffff", fc=ChartEngine._to_matplotlib_rgb(theme_color), alpha=0.9, lw=1.5)
                ax.add_patch(rect)
                ax.text(task['start'] + (task['dur'] / 2), i, f"{task['dur']:g} {unit_str}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
            
            names = [t['task'] for t in tasks]
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_title(title_str, fontsize=13, fontweight='bold', pad=20)
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            max_x = max([t['start'] + t['dur'] for t in tasks])
            ax.set_xlim(0, max_x + (max_x * 0.1))
            ax.set_ylim(-0.6, len(names)-0.4)
            
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception:
            return None

    @staticmethod
    def _professional_palette(theme_color: Tuple[int, int, int]) -> List[str]:
        base = ChartEngine._to_matplotlib_rgb(theme_color)
        accent = '#5B9BD5'
        if sum(theme_color) < 120:
            accent = '#7F7F7F'
        return [accent, '#A5A5A5', '#D9D9D9', '#BFBFBF', '#7F7F7F', '#C9DAF8', '#9EADBA', '#EDEDED']

    @staticmethod
    def _parse_chart_items(raw_data: str) -> List[Tuple[str, float]]:
        items: List[Tuple[str, float]] = []
        for part in raw_data.split(';'):
            tokens = [token.strip() for token in part.split(',')]
            if len(tokens) < 2:
                continue
            label = tokens[0]
            value = re.sub(r'[^\d.\-]', '', tokens[1])
            if not label or not value:
                continue
            try:
                numeric = float(value)
            except ValueError:
                continue
            items.append((label, numeric))
        return items

    @staticmethod
    def create_bar_chart(data_str: str, theme_color: Tuple[int, int, int]) -> Optional[io.BytesIO]:
        try:
            parts = [part.strip() for part in data_str.split('|')]
            if len(parts) >= 3:
                title_str, unit_str, raw_data = parts[0], parts[1], "|".join(parts[2:])
            elif len(parts) == 2:
                title_str, unit_str, raw_data = parts[0], "Nilai", parts[1]
            else:
                title_str, unit_str, raw_data = "Ringkasan", "Nilai", data_str
            items = ChartEngine._parse_chart_items(raw_data)
            if not items:
                return None

            labels = [label for label, _ in items]
            values = [value for _, value in items]
            palette = ChartEngine._professional_palette(theme_color)
            bar_colors = [palette[idx % len(palette)] for idx in range(len(values))]

            fig_height = max(3.6, len(items) * 0.7)
            fig, ax = plt.subplots(figsize=(8.4, fig_height))
            y_positions = list(range(len(items)))
            ax.barh(y_positions, values, color=bar_colors, edgecolor='#666666', linewidth=0.8)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=10)
            ax.invert_yaxis()
            ax.set_title(title_str, fontsize=12, fontweight='bold', pad=14)
            ax.set_xlabel(unit_str, fontsize=10)
            ax.grid(axis='x', linestyle='--', alpha=0.35)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            max_value = max(values) or 1.0
            ax.set_xlim(0, max_value * 1.18)

            for idx, value in enumerate(values):
                ax.text(value + (max_value * 0.02), idx, f"{value:g}", va='center', fontsize=9, color='#404040')

            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception:
            return None

    @staticmethod
    def create_donut_chart(data_str: str, theme_color: Tuple[int, int, int]) -> Optional[io.BytesIO]:
        try:
            parts = [part.strip() for part in data_str.split('|')]
            if len(parts) >= 2:
                title_str, raw_data = parts[0], "|".join(parts[1:])
            else:
                title_str, raw_data = "Komposisi", data_str
            items = ChartEngine._parse_chart_items(raw_data)
            if not items:
                return None

            labels = [label for label, _ in items]
            values = [value for _, value in items]
            palette = ChartEngine._professional_palette(theme_color)
            colors = [palette[idx % len(palette)] for idx in range(len(values))]

            fig, ax = plt.subplots(figsize=(6.6, 4.8))
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                colors=colors,
                startangle=90,
                wedgeprops={'width': 0.45, 'edgecolor': 'white'},
                autopct=lambda pct: f"{pct:.0f}%" if pct >= 8 else '',
                pctdistance=0.8,
                labeldistance=1.05,
            )
            for text in texts:
                text.set_fontsize(9)
            for text in autotexts:
                text.set_fontsize(8.5)
                text.set_color('#404040')
            ax.text(0, 0, "Focus", ha='center', va='center', fontsize=11, fontweight='bold', color='#404040')
            ax.set_title(title_str, fontsize=12, fontweight='bold', pad=12)
            ax.axis('equal')

            img = io.BytesIO()
            plt.tight_layout()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            plt.close()
            img.seek(0)
            return img
        except Exception:
            return None

class DocumentBuilder:
    @staticmethod
    def create_base_document(template_path: str = "") -> Tuple[Document, bool]:
        active_template = str(template_path or "").strip()
        if not active_template or not Path(active_template).exists():
            return Document(), False

        doc = Document(active_template)
        body = doc._element.body
        sect_pr = body.sectPr
        for child in list(body):
            if child is sect_pr:
                continue
            body.remove(child)
        return doc, True

    @staticmethod
    def _coerce_theme_color(theme_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        if not isinstance(theme_color, tuple) or len(theme_color) != 3:
            return DEFAULT_COLOR
        channels = []
        for channel in theme_color:
            try:
                value = int(channel)
            except Exception:
                value = 0
            channels.append(max(0, min(255, value)))
        return tuple(channels)

    @staticmethod
    def _muted_theme_color(theme_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        base = DocumentBuilder._coerce_theme_color(theme_color)
        return tuple(max(25, min(235, int((channel * 0.72) + 24))) for channel in base)

    @staticmethod
    def _set_run_format(
        run,
        size: Optional[float] = None,
        bold: Optional[bool] = None,
        italic: Optional[bool] = None,
        color: Tuple[int, int, int] = StyleEngine.TEXT_COLOR,
        font_name: str = StyleEngine.PROFESSIONAL_FONT,
    ) -> None:
        run.font.name = font_name
        r_pr = run._element.get_or_add_rPr()
        r_fonts = r_pr.find(qn('w:rFonts'))
        if r_fonts is None:
            r_fonts = OxmlElement('w:rFonts')
            r_pr.append(r_fonts)
        r_fonts.set(qn('w:eastAsia'), font_name)
        r_fonts.set(qn('w:ascii'), font_name)
        r_fonts.set(qn('w:hAnsi'), font_name)
        if size is not None:
            run.font.size = Pt(size)
        if bold is not None:
            run.bold = bold
        if italic is not None:
            run.italic = italic
        if color:
            run.font.color.rgb = RGBColor(*color)

    @staticmethod
    def _apply_cell_shading(cell, fill_hex: str) -> None:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = tc_pr.find(qn('w:shd'))
        if shd is None:
            shd = OxmlElement('w:shd')
            tc_pr.append(shd)
        shd.set(qn('w:fill'), fill_hex)

    @staticmethod
    def _set_cell_width(cell, width_inches: float) -> None:
        width_twips = int(max(width_inches, 0.1) * 1440)
        cell.width = Inches(width_inches)
        tc_pr = cell._tc.get_or_add_tcPr()
        tc_w = tc_pr.find(qn('w:tcW'))
        if tc_w is None:
            tc_w = OxmlElement('w:tcW')
            tc_pr.append(tc_w)
        tc_w.set(qn('w:w'), str(width_twips))
        tc_w.set(qn('w:type'), 'dxa')

    @staticmethod
    def _table_column_widths(table) -> Optional[List[float]]:
        if not table.rows:
            return None
        headers = [
            re.sub(r"\s+", " ", cell.text or "").strip().lower()
            for cell in table.rows[0].cells
        ]
        if headers[:6] == ["no", "periode", "fase", "fokus aktivitas", "deliverable utama", "milestone / kontrol"]:
            return [0.32, 0.78, 0.95, 1.45, 1.30, 1.45]
        if len(headers) >= 5 and "periode" in headers and any("deliverable" in header for header in headers):
            return [1.15, 0.90, 1.65, 1.30, 1.25][:len(headers)]
        if len(headers) == 2:
            return [1.55, 4.70]
        if len(headers) == 3:
            return [1.35, 2.45, 2.45]
        if headers[:4] == ["peran tenaga ahli", "fokus tanggung jawab", "kompetensi / pengalaman kunci", "keterlibatan"]:
            return [1.55, 1.65, 2.15, 1.45]
        if len(headers) == 4:
            return [1.35, 1.70, 1.70, 1.50]
        return None

    @staticmethod
    def _format_table(table) -> None:
        table.style = 'Table Grid'
        widths = DocumentBuilder._table_column_widths(table)
        table.autofit = widths is None
        if not table.rows:
            return
        if widths:
            for row in table.rows:
                for idx, cell in enumerate(row.cells):
                    if idx < len(widths):
                        DocumentBuilder._set_cell_width(cell, widths[idx])
        for row_index, row in enumerate(table.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.space_before = Pt(0)
                    paragraph.paragraph_format.space_after = Pt(3)
                    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
                    paragraph.paragraph_format.line_spacing = 1.05
                    for run in paragraph.runs:
                        DocumentBuilder._set_run_format(
                            run,
                            size=9.5 if widths and len(widths) >= 5 else StyleEngine.TABLE_FONT_SIZE,
                            bold=run.bold if row_index != 0 else True,
                        )
                if row_index == 0:
                    DocumentBuilder._apply_cell_shading(cell, StyleEngine.TABLE_HEADER_FILL)

    @staticmethod
    def _set_cell_text(cell, text: str, bold: bool = False) -> None:
        cell.text = ""
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(str(text or "").strip())
        run.bold = bold
        DocumentBuilder._set_run_format(
            run,
            size=StyleEngine.TABLE_FONT_SIZE,
            bold=bold,
            color=StyleEngine.TEXT_COLOR,
        )

    @staticmethod
    def _append_field(paragraph, instruction: str) -> None:
        run = paragraph.add_run()
        field_begin = OxmlElement("w:fldChar")
        field_begin.set(qn("w:fldCharType"), "begin")
        field_instruction = OxmlElement("w:instrText")
        field_instruction.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        field_instruction.text = instruction
        field_separator = OxmlElement("w:fldChar")
        field_separator.set(qn("w:fldCharType"), "separate")
        field_end = OxmlElement("w:fldChar")
        field_end.set(qn("w:fldCharType"), "end")
        run._r.extend([field_begin, field_instruction, field_separator, field_end])

    @staticmethod
    def _enable_update_fields_on_open(doc: Document) -> None:
        settings = doc.settings._element
        update_fields = settings.find(qn("w:updateFields"))
        if update_fields is None:
            update_fields = OxmlElement("w:updateFields")
            settings.append(update_fields)
        update_fields.set(qn("w:val"), "true")

    @staticmethod
    def add_table_of_contents(doc: Document, items: Optional[List[str]] = None) -> None:
        heading = doc.add_paragraph(style="Heading 1")
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        heading.add_run("DAFTAR ISI").bold = True
        toc_paragraph = doc.add_paragraph()
        DocumentBuilder._append_field(toc_paragraph, 'TOC \\o "1-3" \\h \\z \\u')
        for item in items or []:
            clean_item = re.sub(r"\s+", " ", str(item or "")).strip()
            if not clean_item:
                continue
            fallback = doc.add_paragraph(style="Normal")
            fallback.paragraph_format.left_indent = Inches(0.15)
            fallback.paragraph_format.space_after = Pt(2)
            run = fallback.add_run(clean_item)
            DocumentBuilder._set_run_format(run, size=StyleEngine.BODY_FONT_SIZE)
        DocumentBuilder._enable_update_fields_on_open(doc)
        doc.add_page_break()

    @staticmethod
    def add_reference_cover_page(
        doc: Document,
        client: str,
        project: str,
        service_type: str,
        project_type: str,
        timeline: str,
        budget: str,
        firm_profile: Optional[Dict[str, Any]],
        theme_color: Tuple[int, int, int],
        logo_stream: Optional[io.BytesIO] = None,
    ) -> None:
        logo_added = False
        if logo_stream:
            try:
                logo_stream.seek(0)
                cover_logo = doc.add_paragraph()
                cover_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cover_logo.paragraph_format.space_before = Pt(18)
                cover_logo.paragraph_format.space_after = Pt(18)
                cover_logo.add_run().add_picture(logo_stream, width=Inches(3.0))
                logo_added = True
            except (UnrecognizedImageError, OSError, ValueError) as exc:
                logger.warning("Logo skipped due to unsupported image format: %s", exc)
        if not logo_added:
            try:
                fallback_logo = LogoManager._create_fallback_logo(client or "Klien")
                fallback_logo.seek(0)
                cover_logo = doc.add_paragraph()
                cover_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cover_logo.paragraph_format.space_before = Pt(18)
                cover_logo.paragraph_format.space_after = Pt(18)
                cover_logo.add_run().add_picture(fallback_logo, width=Inches(2.4))
                logo_added = True
            except Exception as exc:
                logger.warning("Fallback logo render failed: %s", exc)

        title = doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.paragraph_format.space_before = Pt(54 if not logo_added else 0)
        title.paragraph_format.space_after = Pt(6)
        title_run = title.add_run("PROPOSAL STRATEGIS")
        DocumentBuilder._set_run_format(title_run, size=16, bold=True)

        client_name = doc.add_paragraph()
        client_name.alignment = WD_ALIGN_PARAGRAPH.CENTER
        client_name.paragraph_format.space_after = Pt(12)
        client_run = client_name.add_run((client or "Klien").upper())
        DocumentBuilder._set_run_format(client_run, size=24, bold=True)

        initiative = doc.add_paragraph()
        initiative.alignment = WD_ALIGN_PARAGRAPH.CENTER
        initiative.paragraph_format.space_after = Pt(8)
        initiative_run = initiative.add_run(project or f"{service_type} - {project_type}")
        DocumentBuilder._set_run_format(initiative_run, size=13, italic=True)

        meta_bits = [bit for bit in [service_type, project_type] if str(bit or "").strip()]
        if timeline:
            meta_bits.append(f"Durasi {timeline}")
        if meta_bits:
            meta_line = doc.add_paragraph(" | ".join(meta_bits))
            meta_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
            meta_line.paragraph_format.space_after = Pt(2)
            for run in meta_line.runs:
                DocumentBuilder._set_run_format(run, size=10.5, color=StyleEngine.SUBTLE_TEXT_COLOR)

        if budget:
            budget_line = doc.add_paragraph(f"Estimasi investasi: {budget}")
            budget_line.alignment = WD_ALIGN_PARAGRAPH.CENTER
            budget_line.paragraph_format.space_after = Pt(0)
            for run in budget_line.runs:
                DocumentBuilder._set_run_format(run, size=10.5, color=StyleEngine.SUBTLE_TEXT_COLOR)

        signature = doc.add_paragraph()
        signature.alignment = WD_ALIGN_PARAGRAPH.CENTER
        signature.paragraph_format.space_before = Pt(92)
        signature_run = signature.add_run("Disusun Oleh:")
        DocumentBuilder._set_run_format(signature_run, size=11)
        signature.add_run().add_break()

        firm_run = signature.add_run(WRITER_FIRM_NAME)
        DocumentBuilder._set_run_format(firm_run, size=13, bold=True)

        legal_name = str((firm_profile or {}).get("legal_name") or "").strip()
        if legal_name and legal_name.lower() != WRITER_FIRM_NAME.lower():
            signature.add_run().add_break()
            legal_run = signature.add_run(legal_name)
            DocumentBuilder._set_run_format(legal_run, size=10, color=StyleEngine.SUBTLE_TEXT_COLOR)

        doc.add_page_break()

    @staticmethod
    def add_reference_chapter_heading(
        doc: Document,
        chapter_title: str,
        theme_color: Tuple[int, int, int],
        is_first_chapter: bool = False,
    ) -> None:
        try:
            heading = doc.add_paragraph(style="Heading 1")
        except KeyError:
            heading = doc.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        heading.paragraph_format.space_before = Pt(0)
        heading.paragraph_format.space_after = Pt(10)
        heading.paragraph_format.keep_with_next = True
        run = heading.add_run(chapter_title)
        DocumentBuilder._set_run_format(run, size=14, bold=True)

    @staticmethod
    def _compact_words(value: Any, max_words: int = 44) -> str:
        text = clean_markup_artifacts(value)
        text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
        text = re.sub(r"\s+", " ", text).strip(" .;:")
        if not text:
            return ""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).rstrip(" ,;:") + "."

    @staticmethod
    def _certification_summary(value: Any, fallback: Any = "") -> str:
        primary = clean_markup_artifacts(value)
        primary = re.sub(r"(?i)\blihat\s+website\s+resmi\s+untuk\s+daftar\s+lengkap\b", "", primary)
        fallback_text = clean_markup_artifacts(fallback)
        text = " ".join(part for part in [primary, fallback_text] if part).strip()
        text = re.sub(r"(?i)\blihat\s+website\s+resmi\s+untuk\s+daftar\s+lengkap\b", "", text)
        known_patterns = [
            r"Lead\s+Auditor\s+ISO\s*27001", r"ISO/IEC\s*27001", r"ISO\s*27001",
            r"ISO\s*20000", r"TOGAF(?:\s*9\s*Foundations)?", r"COBIT\s*5?",
            r"ITIL(?:\s*(?:Foundation|Service Strategy|V3|Certificate))*", r"CAPM",
            r"CompTIA\s*Project\+", r"Project\+", r"CEH", r"CHFI", r"CISA",
            r"CCNA(?:\s*Routing\s*and\s*Switching)?", r"ECIH", r"EDRP",
        ]
        found: List[str] = []
        seen = set()
        for pattern in known_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                item = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;:")
                key = item.lower()
                if item and key not in seen:
                    seen.add(key)
                    found.append(item)
        if found:
            return "Kredensial yang dapat ditonjolkan mencakup " + ", ".join(found[:10]) + "."
        return DocumentBuilder._compact_words(text, max_words=34)

    @staticmethod
    def _capability_without_names(value: Any) -> str:
        text = clean_markup_artifacts(value)
        lowered = text.lower()
        domains: List[str] = []
        for label, tokens in [
            ("manajemen proyek", ["capm", "project+", "project manager", "pmo"]),
            ("arsitektur enterprise dan tata kelola TI", ["togaf", "cobit", "arsitektur"]),
            ("IT service management", ["itil", "iso 20000", "service management"]),
            ("keamanan informasi dan audit", ["iso 27001", "ceh", "chfi", "cisa", "lead auditor", "ccna"]),
            ("pemulihan dan risiko operasional", ["edrp", "disaster recovery", "risk"]),
        ]:
            if any(token in lowered for token in tokens) and label not in domains:
                domains.append(label)
        if domains:
            return "Kapabilitas internal mencakup " + ", ".join(domains[:5]) + ", dengan peran delivery yang dipetakan sesuai kebutuhan proyek."
        return DocumentBuilder._compact_words(text, max_words=32)

    @staticmethod
    def add_writer_firm_profile_section(
        doc: Document,
        firm_profile: Optional[Dict[str, Any]],
        theme_color: Tuple[int, int, int],
    ) -> None:
        profile = firm_profile or {}
        summary = str(profile.get("profile_summary") or "").strip()
        portfolio = str(profile.get("portfolio_highlights") or "").strip()
        credentials = str(profile.get("credential_highlights") or "").strip()
        legal_name = str(profile.get("legal_name") or "").strip()
        values_approach = str(profile.get("values_approach") or "").strip()
        team_expertise = str(profile.get("team_expertise") or "").strip()
        portfolio_scale = str(profile.get("portfolio_scale") or "").strip()
        certifications = str(profile.get("certifications") or "").strip()
        accolades = str(profile.get("accolades") or "").strip()

        contact_rows = [
            ("Alamat kantor", profile.get("office_address", "")),
            ("Email", profile.get("email", "")),
            ("Telp", profile.get("phone", "")),
            ("WhatsApp", profile.get("whatsapp", "")),
            ("Website", profile.get("website", "")),
            ("Jam operasional", profile.get("operating_hours", "")),
        ]
        detail_rows = [
            ("Entitas hukum", legal_name),
            ("Fokus layanan", "Pelatihan IT, sertifikasi, dan konsultasi IT."),
            ("Portofolio", DocumentBuilder._compact_words(portfolio, max_words=38)),
            ("Kapabilitas", DocumentBuilder._capability_without_names(credentials or team_expertise)),
            ("Pendekatan berbasis sumber publik", DocumentBuilder._compact_words(values_approach, max_words=42)),
            ("Keahlian tim berbasis sumber publik", DocumentBuilder._compact_words(team_expertise, max_words=34)),
            ("Skala dan bukti portofolio", DocumentBuilder._compact_words(portfolio_scale, max_words=34)),
            ("Sertifikasi dan kredensial eksternal", DocumentBuilder._certification_summary(certifications, fallback=credentials)),
            ("Pengakuan eksternal", DocumentBuilder._compact_words(accolades, max_words=28)),
        ]
        visible_detail_rows = [(label, str(value or "").strip()) for label, value in detail_rows if str(value or "").strip()]
        visible_contact_rows = [(label, str(value or "").strip()) for label, value in contact_rows if str(value or "").strip()]

        if not summary and not visible_detail_rows and not visible_contact_rows:
            return

        StyleEngine.add_horizontal_line(doc, color=(214, 220, 228))

        try:
            heading = doc.add_paragraph(style="Heading 2")
        except KeyError:
            heading = doc.add_paragraph()
        heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
        heading.paragraph_format.space_before = Pt(8)
        heading.paragraph_format.space_after = Pt(8)
        heading_run = heading.add_run("Kapabilitas Konsultan dan Kredensial Pendukung")
        DocumentBuilder._set_run_format(heading_run, size=12, bold=True)

        if summary:
            summary_paragraph = doc.add_paragraph(summary)
            summary_paragraph.paragraph_format.space_after = Pt(8)

        if visible_detail_rows:
            details_table = doc.add_table(rows=1, cols=2)
            DocumentBuilder._set_cell_text(details_table.rows[0].cells[0], "Aspek", bold=True)
            DocumentBuilder._set_cell_text(details_table.rows[0].cells[1], "Keterangan", bold=True)
            for label, value in visible_detail_rows:
                row = details_table.add_row().cells
                DocumentBuilder._set_cell_text(row[0], label, bold=True)
                DocumentBuilder._set_cell_text(row[1], value)
            DocumentBuilder._format_table(details_table)

        if visible_contact_rows:
            contact_title = doc.add_paragraph()
            contact_title.paragraph_format.space_before = Pt(8)
            contact_title.paragraph_format.space_after = Pt(6)
            run = contact_title.add_run("Kontak Resmi")
            DocumentBuilder._set_run_format(run, size=11, bold=True)

            contact_table = doc.add_table(rows=1, cols=2)
            DocumentBuilder._set_cell_text(contact_table.rows[0].cells[0], "Kanal", bold=True)
            DocumentBuilder._set_cell_text(contact_table.rows[0].cells[1], "Detail", bold=True)
            for label, value in visible_contact_rows:
                row = contact_table.add_row().cells
                DocumentBuilder._set_cell_text(row[0], label, bold=True)
                DocumentBuilder._set_cell_text(row[1], value)
            DocumentBuilder._format_table(contact_table)

        source_urls = profile.get("official_source_urls") or []
        if isinstance(source_urls, str):
            source_urls = [item.strip() for item in source_urls.split(",") if item.strip()]
        domains = []
        seen_domains = set()
        for url in source_urls:
            host = urlparse(str(url or "")).netloc.replace("www.", "").strip()
            if not host or host in seen_domains:
                continue
            seen_domains.add(host)
            domains.append(host)
        if domains:
            note = doc.add_paragraph()
            note.paragraph_format.space_before = Pt(6)
            note.paragraph_format.space_after = Pt(0)
            note_run = note.add_run(
                "Profil dan kontak pada bagian ini dirangkum dari kanal resmi yang terverifikasi: "
                + ", ".join(domains)
                + "."
            )
            DocumentBuilder._set_run_format(note_run, size=9, italic=True, color=StyleEngine.SUBTLE_TEXT_COLOR)

    @staticmethod
    def _append_text_run(paragraph, text: str, bold: bool = False, italic: bool = False) -> None:
        cleaned = re.sub(r'\s+', ' ', text or '').strip()
        if not cleaned:
            return

        if paragraph.runs:
            last_text = paragraph.runs[-1].text or ""
            if last_text and not last_text.endswith((" ", "\t", "\n", "(", "[", "/")) and not cleaned.startswith((".", ",", ";", ":", ")", "]", "%")):
                cleaned = " " + cleaned

        run = paragraph.add_run(cleaned)
        DocumentBuilder._set_run_format(run, size=StyleEngine.BODY_FONT_SIZE, bold=bold, italic=italic)

    @staticmethod
    def _style_num_id(doc: Document, style_name: str) -> Optional[int]:
        try:
            style = doc.styles[style_name]
        except KeyError:
            return None
        ppr = style.element.pPr
        if ppr is None or ppr.numPr is None or ppr.numPr.numId is None:
            return None
        try:
            return int(ppr.numPr.numId.val)
        except Exception:
            return None

    @staticmethod
    def _create_list_num_id(doc: Document, style_name: str) -> Optional[int]:
        style_num_id = DocumentBuilder._style_num_id(doc, style_name)
        if style_num_id is None:
            return None

        numbering = doc.part.numbering_part.numbering_definitions._numbering
        abstract_num_id = None
        for num in numbering.findall(qn('w:num')):
            raw_num_id = num.get(qn('w:numId'))
            try:
                current_num_id = int(raw_num_id) if raw_num_id is not None else None
            except (TypeError, ValueError):
                current_num_id = None
            if current_num_id != style_num_id:
                continue
            abstract = num.find(qn('w:abstractNumId'))
            if abstract is None:
                continue
            raw_abstract = abstract.get(qn('w:val'))
            try:
                abstract_num_id = int(raw_abstract) if raw_abstract is not None else None
            except (TypeError, ValueError):
                abstract_num_id = None
            break

        if abstract_num_id is None:
            return None

        existing_num_ids: List[int] = []
        for num in numbering.findall(qn('w:num')):
            raw_num_id = num.get(qn('w:numId'))
            try:
                if raw_num_id is not None:
                    existing_num_ids.append(int(raw_num_id))
            except (TypeError, ValueError):
                continue
        next_num_id = (max(existing_num_ids) + 1) if existing_num_ids else (style_num_id + 1)

        num = OxmlElement('w:num')
        num.set(qn('w:numId'), str(next_num_id))
        abstract_ref = OxmlElement('w:abstractNumId')
        abstract_ref.set(qn('w:val'), str(abstract_num_id))
        num.append(abstract_ref)
        lvl_override = OxmlElement('w:lvlOverride')
        lvl_override.set(qn('w:ilvl'), '0')
        start_override = OxmlElement('w:startOverride')
        start_override.set(qn('w:val'), '1')
        lvl_override.append(start_override)
        num.append(lvl_override)
        numbering.append(num)
        return next_num_id

    @staticmethod
    def _apply_list_num_id(paragraph, num_id: int, level: int = 0) -> None:
        p = paragraph._p
        ppr = p.get_or_add_pPr()
        num_pr = ppr.find(qn('w:numPr'))
        if num_pr is None:
            num_pr = OxmlElement('w:numPr')
            ppr.append(num_pr)

        ilvl = num_pr.find(qn('w:ilvl'))
        if ilvl is None:
            ilvl = OxmlElement('w:ilvl')
            num_pr.append(ilvl)
        ilvl.set(qn('w:val'), str(level))

        num_id_el = num_pr.find(qn('w:numId'))
        if num_id_el is None:
            num_id_el = OxmlElement('w:numId')
            num_pr.append(num_id_el)
        num_id_el.set(qn('w:val'), str(num_id))

    @staticmethod
    def _format_plain_paragraph(paragraph) -> None:
        pf = paragraph.paragraph_format
        pf.left_indent = Pt(0)
        pf.first_line_indent = Pt(0)
        pf.space_before = Pt(0)
        pf.space_after = Pt(StyleEngine.BODY_SPACE_AFTER)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = StyleEngine.BODY_LINE_SPACING

    @staticmethod
    def _format_list_paragraph(paragraph, level: int = 0) -> None:
        safe_level = max(0, min(int(level or 0), 8))
        pf = paragraph.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(5)
        pf.left_indent = Inches(0.36 + (0.26 * safe_level))
        pf.first_line_indent = Inches(-0.18)
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = 1.15

    @staticmethod
    def _paragraph_list_level(paragraph) -> int:
        try:
            num_pr = paragraph._p.get_or_add_pPr().find(qn('w:numPr'))
            ilvl = num_pr.find(qn('w:ilvl')) if num_pr is not None else None
            return int(ilvl.get(qn('w:val'))) if ilvl is not None else 0
        except Exception:
            return 0

    @staticmethod
    def _render_html_list(doc: Document, element, level: int = 0) -> None:
        safe_level = max(0, min(int(level or 0), 8))
        direct_items = element.find_all('li', recursive=False)
        is_ordered_list = element.name == 'ol'
        style_name = 'List Number' if is_ordered_list else 'List Bullet'
        list_num_id = DocumentBuilder._create_list_num_id(doc, style_name)
        for idx, li in enumerate(direct_items, start=1):
            inline_text = " ".join(
                str(child).strip()
                for child in li.contents
                if getattr(child, "name", None) not in {"ul", "ol"} and str(child).strip()
            )
            nested_lists = li.find_all(['ul', 'ol'], recursive=False)
            if not inline_text and not nested_lists:
                continue
            try:
                p = doc.add_paragraph(style=style_name)
            except KeyError:
                p = doc.add_paragraph()
                fallback_marker = f"{idx}. " if is_ordered_list else "- "
                marker_run = p.add_run(fallback_marker)
                marker_run.bold = True
            if list_num_id is not None:
                DocumentBuilder._apply_list_num_id(p, list_num_id, level=safe_level)
            DocumentBuilder._format_list_paragraph(p, level=safe_level)
            DocumentBuilder._process_inline_html(p, li, skip_nested_lists=True)
            for nested in nested_lists:
                DocumentBuilder._render_html_list(doc, nested, level=safe_level + 1)

    @staticmethod
    def parse_html_to_docx(doc: Document, html_content: str, theme_color: Tuple[int, int, int]) -> None:
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup.children:
            if element.name is None: continue
            if element.name in ['h1', 'h2', 'h3']:
                level = int(element.name[1])
                p = doc.add_heading(element.get_text().strip(), level=level)
                if level == 1:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    DocumentBuilder._set_run_format(
                        run,
                        size=StyleEngine.HEADING_1_SIZE if level == 1 else StyleEngine.HEADING_2_SIZE if level == 2 else StyleEngine.HEADING_3_SIZE,
                        bold=True,
                    )
            elif element.name == 'p':
                if len(element.contents) == 1 and getattr(element.contents[0], "name", None) == "code":
                    code_text = element.contents[0].get_text("\n", strip=False)
                    if DocumentBuilder._is_visual_code_block(code_text):
                        DocumentBuilder._render_visual_code_block(doc, code_text)
                        continue
                p = doc.add_paragraph()
                DocumentBuilder._format_plain_paragraph(p)
                DocumentBuilder._process_inline_html(p, element)
            elif element.name == 'pre':
                code_element = element.find('code')
                code_text = code_element.get_text("\n", strip=False) if code_element is not None else element.get_text("\n", strip=False)
                if DocumentBuilder._is_visual_code_block(code_text):
                    DocumentBuilder._render_visual_code_block(doc, code_text)
                    continue
                p = doc.add_paragraph()
                DocumentBuilder._format_plain_paragraph(p)
                p.add_run(code_text)
            elif element.name in ['ul', 'ol']:
                DocumentBuilder._render_html_list(doc, element, level=0)
            elif element.name == 'table':
                rows = element.find_all('tr')
                if not rows: continue
                max_cols = max([len(r.find_all(['td', 'th'])) for r in rows])
                table = doc.add_table(rows=len(rows), cols=max_cols)
                for i, row in enumerate(rows):
                    cols = row.find_all(['td', 'th'])
                    for j, col in enumerate(cols):
                        if j < max_cols:
                            cell = table.cell(i, j)
                            cell._element.clear_content()
                            p = cell.add_paragraph()
                            DocumentBuilder._process_inline_html(p, col)
                            if col.name == 'th' or i == 0:
                                for run in p.runs:
                                    run.bold = True
                DocumentBuilder._format_table(table)

    @staticmethod
    def _process_inline_html(paragraph, element, skip_nested_lists: bool = False):
        for child in element.children:
            if skip_nested_lists and getattr(child, "name", None) in ['ul', 'ol']:
                continue
            if child.name in ['strong', 'b']:
                DocumentBuilder._append_text_run(paragraph, child.get_text(" ", strip=True), bold=True)
            elif child.name in ['em', 'i']:
                DocumentBuilder._append_text_run(paragraph, child.get_text(" ", strip=True), italic=True)
            elif child.name == 'br':
                paragraph.add_run("\n")
            elif child.name is None:
                DocumentBuilder._append_text_run(paragraph, clean_markup_artifacts(str(child)))
            else:
                DocumentBuilder._process_inline_html(paragraph, child, skip_nested_lists=skip_nested_lists)

    @staticmethod
    def _is_visual_code_block(text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        if re.search(r"[█▓▒░▇▆▅▄▃▂▁■□▰▱]", cleaned):
            return True
        return bool(re.search(r"\b(visual|proyeksi|roadmap|jadwal|alokasi|anggaran|week|minggu)\b", cleaned, re.IGNORECASE))

    @staticmethod
    def _summarize_visual_bar(value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            return ""

        bar_chars = "█▓▒░▇▆▅▄▃▂▁■□▰▱"
        filtered = [char for char in text if not char.isspace()]
        if filtered and all(char in bar_chars for char in filtered):
            total = len(filtered)
            filled = sum(1 for char in filtered if char in "█▓▒▇▆▅▄▃▂▁■▰")
            if total > 0:
                return f"{round((filled / total) * 100):.0f}% selesai"

        remaining = re.sub(r"[█▓▒░▇▆▅▄▃▂▁■□▰▱]+", " ", text)
        remaining = re.sub(r"\s+", " ", remaining).strip(" -;,:")
        if remaining:
            return remaining
        return text

    @staticmethod
    def _render_visual_code_block(doc: Document, raw_text: str) -> bool:
        lines = [re.sub(r"\s+", " ", line).strip() for line in (raw_text or "").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return False

        title = lines[0]
        body_lines = lines[1:]
        if not body_lines and ":" not in title:
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_after = Pt(6)
            run = paragraph.add_run(title)
            DocumentBuilder._set_run_format(run, size=11, bold=True)
            return True

        if title:
            title_paragraph = doc.add_paragraph()
            title_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            title_paragraph.paragraph_format.space_after = Pt(4)
            title_run = title_paragraph.add_run(title)
            DocumentBuilder._set_run_format(title_run, size=11, bold=True)

        rows: List[Tuple[str, str]] = []
        plain_lines: List[str] = []
        for line in body_lines:
            if ":" in line:
                label, value = line.split(":", 1)
                rows.append((label.strip(), DocumentBuilder._summarize_visual_bar(value)))
            else:
                plain_lines.append(line)

        for line in plain_lines:
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.space_after = Pt(3)
            run = paragraph.add_run(line)
            DocumentBuilder._set_run_format(run, size=StyleEngine.BODY_FONT_SIZE)

        if rows:
            table = doc.add_table(rows=1, cols=2)
            DocumentBuilder._set_cell_text(table.rows[0].cells[0], "Bagian", bold=True)
            DocumentBuilder._set_cell_text(table.rows[0].cells[1], "Detail", bold=True)
            for label, value in rows:
                cells = table.add_row().cells
                DocumentBuilder._set_cell_text(cells[0], label, bold=True)
                DocumentBuilder._set_cell_text(cells[1], value)
            DocumentBuilder._format_table(table)
        return True

    @staticmethod
    def _markdown_table_cells(line: str) -> List[str]:
        stripped = (line or "").strip()
        if not stripped.startswith("|"):
            return []
        return [cell.strip() for cell in stripped.strip("|").split("|")]

    @staticmethod
    def _is_markdown_table_separator(line: str) -> bool:
        cells = DocumentBuilder._markdown_table_cells(line)
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)

    @staticmethod
    def _is_valid_markdown_table_block(lines: List[str]) -> bool:
        if len(lines) < 3 or not DocumentBuilder._is_markdown_table_separator(lines[1]):
            return False
        header = DocumentBuilder._markdown_table_cells(lines[0])
        separator = DocumentBuilder._markdown_table_cells(lines[1])
        body_rows = [DocumentBuilder._markdown_table_cells(line) for line in lines[2:]]
        if len(header) < 2 or len(separator) != len(header):
            return False
        return any(len(row) == len(header) and any(cell for cell in row) for row in body_rows)

    @staticmethod
    def _demote_invalid_markdown_tables(raw_text: str) -> str:
        output: List[str] = []
        pending: List[str] = []

        def flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            if DocumentBuilder._is_valid_markdown_table_block(pending):
                output.extend(DocumentBuilder._dedupe_markdown_table_rows(pending))
            else:
                repaired = DocumentBuilder._repair_markdown_table_block(pending)
                if repaired:
                    output.extend(DocumentBuilder._dedupe_markdown_table_rows(repaired))
                else:
                    for row in pending:
                        if DocumentBuilder._is_markdown_table_separator(row):
                            continue
                        cells = [cell for cell in DocumentBuilder._markdown_table_cells(row) if cell]
                        demoted = "; ".join(cells).strip()
                        if demoted:
                            output.append(demoted)
            pending = []

        for raw_line in (raw_text or "").splitlines():
            line = raw_line.strip()
            if line.startswith("|"):
                pending.append(line)
                continue
            flush_pending()
            output.append(raw_line)
        flush_pending()
        return "\n".join(output).strip()

    @staticmethod
    def _repair_markdown_table_block(lines: List[str]) -> List[str]:
        if len(lines) < 3 or not DocumentBuilder._is_markdown_table_separator(lines[1]):
            return []
        header = DocumentBuilder._markdown_table_cells(lines[0])
        body_rows = [DocumentBuilder._markdown_table_cells(line) for line in lines[2:]]
        if len(header) < 2 or not body_rows:
            return []
        body_lengths = [len(row) for row in body_rows if any(cell for cell in row)]
        if not body_lengths:
            return []
        max_len = max(body_lengths)
        if max_len <= len(header) or max_len > 8:
            return []
        expanded_header = [*header, *[f"Detail Tambahan {idx}" for idx in range(1, max_len - len(header) + 1)]]
        repaired = [
            "| " + " | ".join(expanded_header) + " |",
            "| " + " | ".join(["---"] * max_len) + " |",
        ]
        for row in body_rows:
            if not any(cell for cell in row):
                continue
            padded = [*row[:max_len], *([""] * max(0, max_len - len(row)))]
            repaired.append("| " + " | ".join(padded) + " |")
        return repaired

    @staticmethod
    def _dedupe_markdown_table_rows(lines: List[str]) -> List[str]:
        if len(lines) < 3:
            return lines
        output = lines[:2]
        seen = set()
        for row in lines[2:]:
            cells = DocumentBuilder._markdown_table_cells(row)
            signature = tuple(re.sub(r"\s+", " ", cell).strip().lower() for cell in cells)
            if signature in seen:
                continue
            seen.add(signature)
            output.append(row)
        return output

    @staticmethod
    def _normalize_markdown_blocks(raw_text: str) -> str:
        normalized: List[str] = []
        ordered_pattern = re.compile(r'^\d+[.)]\s+')
        bullet_pattern = re.compile(r'^[-*•▪◦●]\s+')

        for raw_line in (raw_text or "").split('\n'):
            leading = re.match(r'^(\s*)', raw_line).group(1)
            stripped = raw_line.strip()
            is_indented_list = bool(re.match(r'^(\d+[.)]|[-*•▪◦●])\s+', stripped))
            line = f"{leading}{stripped}" if is_indented_list else stripped
            line_for_match = line.strip()
            if re.match(r'^\d+\)\s+', line_for_match):
                normalized_ordered = re.sub(r'^(\d+)\)\s+', r'\1. ', line_for_match)
                line = f"{leading}{normalized_ordered}"
                line_for_match = line.strip()
            if re.match(r'^[-*•▪◦●]\s+', line_for_match):
                normalized_bullet = re.sub(r'^[-*•▪◦●]\s+', '- ', line_for_match)
                line = f"{leading}{normalized_bullet}"
                line_for_match = line.strip()
            previous = normalized[-1].strip() if normalized else ""
            is_ordered = bool(ordered_pattern.match(line_for_match))
            is_bullet = bool(bullet_pattern.match(line_for_match))
            is_list = is_ordered or is_bullet
            previous_is_ordered = bool(ordered_pattern.match(previous))
            previous_is_bullet = bool(bullet_pattern.match(previous))
            previous_is_list = previous_is_ordered or previous_is_bullet
            if is_bullet and previous_is_ordered and not leading:
                line = f"    {line_for_match}"
                leading = "    "
            is_nested_list = bool(is_list and leading)
            is_table = line_for_match.startswith('|')
            is_heading = line_for_match.startswith('#')
            is_visual = line_for_match.startswith('[[') and line_for_match.endswith(']]')

            if is_list and previous and (
                (
                    not previous_is_list
                    and not previous.startswith('|')
                    and not previous.startswith('#')
                    and not previous.startswith('[[')
                )
                or (
                    previous_is_list
                    and not is_nested_list
                    and (is_ordered != previous_is_ordered or is_bullet != previous_is_bullet)
                )
            ):
                normalized.append("")
            if line_for_match and previous_is_list and not is_list and not is_table and not is_heading and not is_visual:
                normalized.append("")
            normalized.append(line)

        compacted: List[str] = []
        blank_streak = 0
        for line in normalized:
            if line:
                blank_streak = 0
                compacted.append(line)
                continue
            blank_streak += 1
            if blank_streak <= 1:
                compacted.append("")
        return "\n".join(compacted).strip()

    @staticmethod
    def process_content(doc: Document, raw_text: str, theme_color: Tuple[int, int, int], chapter_title: str) -> None:
        clean_lines = []
        in_table = False
        normalized_text = sanitize_reader_facing_sources(raw_text)
        normalized_text = DocumentBuilder._normalize_markdown_blocks(clean_markup_artifacts(normalized_text))
        normalized_text = DocumentBuilder._demote_invalid_markdown_tables(normalized_text)
        for raw_line in normalized_text.split('\n'):
            line = raw_line.rstrip()
            stripped_line = line.strip()
            if stripped_line.startswith('[[GANTT:') and stripped_line.endswith(']]'):
                data = stripped_line.replace('[[GANTT:', '').replace(']]', '').strip()
                img = ChartEngine.create_gantt_chart(data, theme_color)
                if img:
                    paragraph = doc.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    paragraph.paragraph_format.space_after = Pt(8)
                    paragraph.add_run().add_picture(img, width=Inches(6.1))
                continue
            if stripped_line.startswith('[[BAR:') and stripped_line.endswith(']]'):
                data = stripped_line.replace('[[BAR:', '').replace(']]', '').strip()
                img = ChartEngine.create_bar_chart(data, theme_color)
                if img:
                    paragraph = doc.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    paragraph.paragraph_format.space_after = Pt(8)
                    paragraph.add_run().add_picture(img, width=Inches(6.1))
                continue
            if stripped_line.startswith('[[DONUT:') and stripped_line.endswith(']]'):
                data = stripped_line.replace('[[DONUT:', '').replace(']]', '').strip()
                img = ChartEngine.create_donut_chart(data, theme_color)
                if img:
                    paragraph = doc.add_paragraph()
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    paragraph.paragraph_format.space_after = Pt(8)
                    paragraph.add_run().add_picture(img, width=Inches(5.6))
                continue
            if stripped_line.startswith('|'):
                if not in_table and clean_lines and clean_lines[-1] != "":
                    clean_lines.append("")
                in_table = True
                clean_lines.append(stripped_line)
            else:
                if in_table and stripped_line and clean_lines and clean_lines[-1] != "":
                    clean_lines.append("")
                in_table = False
                clean_lines.append(line)
            
        html = markdown.markdown("\n".join(clean_lines), extensions=['tables', 'sane_lists'])
        DocumentBuilder.parse_html_to_docx(doc, html, theme_color)

    @staticmethod
    def _paragraph_is_blank(paragraph_el) -> bool:
        if paragraph_el.find('.//' + qn('w:drawing')) is not None:
            return False
        if paragraph_el.find('.//' + qn('w:pBdr')) is not None:
            return False
        for br in paragraph_el.findall('.//' + qn('w:br')):
            if br.get(qn('w:type')) == 'page':
                return False
        texts = [
            node.text or ""
            for node in paragraph_el.findall('.//' + qn('w:t'))
        ]
        return not "".join(texts).strip()

    @staticmethod
    def _paragraph_is_page_break_only(paragraph_el) -> bool:
        if paragraph_el.find('.//' + qn('w:drawing')) is not None:
            return False
        texts = [node.text or "" for node in paragraph_el.findall('.//' + qn('w:t'))]
        if "".join(texts).strip():
            return False
        breaks = paragraph_el.findall('.//' + qn('w:br'))
        return bool(breaks) and all(br.get(qn('w:type')) == 'page' for br in breaks)

    @staticmethod
    def compact_layout(doc: Document) -> None:
        body = doc._element.body
        for child in list(body):
            if child.tag == qn('w:p') and DocumentBuilder._paragraph_is_blank(child):
                body.remove(child)

        children = list(body)
        for idx, child in enumerate(children):
            if child.tag != qn('w:p') or not DocumentBuilder._paragraph_is_page_break_only(child):
                continue

            previous_substantive = None
            for prev in reversed(children[:idx]):
                if prev.tag == qn('w:p') and DocumentBuilder._paragraph_is_blank(prev):
                    continue
                previous_substantive = prev
                break

            next_substantive = None
            for nxt in children[idx + 1:]:
                if nxt.tag == qn('w:p') and DocumentBuilder._paragraph_is_blank(nxt):
                    continue
                next_substantive = nxt
                break

            previous_is_page_break = bool(
                previous_substantive is not None
                and previous_substantive.tag == qn('w:p')
                and DocumentBuilder._paragraph_is_page_break_only(previous_substantive)
            )
            next_is_page_break = bool(
                next_substantive is not None
                and next_substantive.tag == qn('w:p')
                and DocumentBuilder._paragraph_is_page_break_only(next_substantive)
            )
            next_is_real_content = bool(
                next_substantive is not None
                and not next_is_page_break
                and (
                    next_substantive.tag != qn('w:p')
                    or not DocumentBuilder._paragraph_is_blank(next_substantive)
                )
            )

            if previous_is_page_break or next_is_page_break or not next_is_real_content:
                body.remove(child)

        for paragraph in doc.paragraphs:
            style_name = str(getattr(paragraph.style, "name", "") or "")
            if style_name.startswith("Heading"):
                paragraph.paragraph_format.keep_with_next = True
                continue
            if paragraph.alignment == WD_ALIGN_PARAGRAPH.CENTER:
                continue
            if paragraph._element.find('.//' + qn('w:drawing')) is not None:
                continue
            if style_name in {"List Bullet", "List Number"}:
                DocumentBuilder._format_list_paragraph(
                    paragraph,
                    level=DocumentBuilder._paragraph_list_level(paragraph),
                )
            else:
                DocumentBuilder._format_plain_paragraph(paragraph)
