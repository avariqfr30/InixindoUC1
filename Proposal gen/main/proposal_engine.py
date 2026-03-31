"""Proposal orchestration logic: drafting, tightening, compression, and document assembly."""

from .proposal_shared import *
from .runtime_components import ChartEngine, DocumentBuilder, FinancialAnalyzer, FirmAPIClient, LogoManager, StyleEngine, Researcher
from .proposal_support import ProposalSupportMixin


class ProposalEngineMixin:
    def _chapter_generation_options(self, target_words: int, purpose: str = "draft") -> Dict[str, Any]:
        throughput = self._throughput_mode()
        if purpose == "tighten":
            cap = 1800 if throughput else 2400
            floor = 850
            multiplier = 1.75 if throughput else 1.95
            temperature = 0.12 if throughput else 0.15
        elif purpose == "retry":
            cap = 2800 if throughput else 3600
            floor = 1200
            multiplier = 2.15 if throughput else 2.35
            temperature = 0.18 if throughput else 0.2
        else:
            cap = 2800 if throughput else 3600
            floor = 1350
            multiplier = 2.2 if throughput else 2.45
            temperature = 0.2 if throughput else 0.25
        return {
            "num_ctx": 32768 if throughput else 65536,
            "num_predict": max(floor, min(cap, int(target_words * multiplier))),
            "temperature": temperature,
            "top_p": 0.85,
            "repeat_penalty": 1.1,
        }

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    @staticmethod
    def _enhance_closing_with_firm_osint(
        closing_content: str,
        firm_profile: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhance closing chapter with comprehensive firm information via OSINT.
        Appends firm details such as team expertise, certifications, and accolades
        to the closing chapter.
        
        Args:
            closing_content: Existing closing chapter content
            firm_profile: Optional firm profile dictionary
        
        Returns:
            Enhanced closing content with firm information
        """
        try:
            # Use consolidated ProposalSupportMixin method
            return ProposalSupportMixin._enhance_closing_with_firm_details(
                closing_content,
                WRITER_FIRM_NAME,
                firm_profile,
                firm_profile.get("office_address", "") if firm_profile else ""
            )
        except Exception as e:
            logger.warning(f"Error enhancing closing with firm OSINT: {e}")
            return closing_content

    @staticmethod
    def _chapter_excerpt(text: str, max_words: int = 170) -> str:
        words = re.findall(r"\S+", text)
        if len(words) <= max_words:
            return " ".join(words)
        head = " ".join(words[:110])
        tail = " ".join(words[-60:])
        return f"{head} ... {tail}"

    def _apply_global_coherence(
        self,
        chapter_outputs: Dict[str, str],
        selected_chapters: List[Dict[str, Any]],
        client: str,
        project: str
    ) -> Dict[str, str]:
        return chapter_outputs

    def _draft_chapter(
        self,
        chapter: Dict[str, Any],
        prompt: str,
        client: str,
        target_words: int,
        allowed_external_citations: Optional[Set[str]] = None,
        personalization_pack: Optional[Dict[str, Any]] = None
    ) -> str:
        allowed = set(allowed_external_citations or set())
        # First draft pass for a chapter.
        res = self.ollama.chat(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': (
                    f"Tulis konten untuk {chapter['title']} dalam satu draft final. "
                    "Pastikan hard checks terpenuhi: H2 wajib lengkap, word range sesuai target, "
                    "ada numbered list dan bullet list, serta konten tetap konkret dan action-oriented."
                )}
            ],
            options=self._chapter_generation_options(target_words, purpose="draft")
        )
        content = self._apply_draft_repairs(
            chapter=chapter,
            content=(res.get('message', {}).get('content', '') or '').strip(),
            client=client,
            allowed_external_citations=allowed,
            personalization_pack=personalization_pack
        )
        report = self._evaluate_chapter_quality(
            chapter,
            content,
            client,
            target_words=target_words,
            allowed_external_citations=allowed,
            personalization_pack=personalization_pack
        )
        hard_check_keys = {
            "missing_h2", "too_short", "too_long",
            "list_structure", "missing_visual", "citation_policy", "missing_personalization"
        }
        if self._throughput_mode():
            hard_check_keys.discard("too_long")
            hard_check_keys.discard("list_structure")
            hard_check_keys.discard("missing_personalization")
        hard_issues = [i for i in report["issues"] if i in hard_check_keys]
        if not hard_issues:
            return content

        # Retry once when hard checks fail.
        citation_policy_note = (
            f"Allowed external citations: {', '.join(sorted(allowed)) if allowed else 'none'}.\n"
            f"Invalid external citations found: {', '.join(report.get('invalid_external_citations', [])) or '-'}.\n"
            "Hapus semua sitasi eksternal yang tidak ada di daftar allowed.\n"
        )
        personalization_note = (
            f"Sinyal personalisasi yang belum muncul: "
            f"{', '.join(report.get('missing_personalization_signals', [])) or '-'}.\n"
            "Pastikan konten mencerminkan konteks klien, KPI tailoring, terminologi domain, dan anchor inisiatif.\n"
        )
        retry_prompt = (
            f"Perbaiki draft {chapter['title']} agar lulus hard quality checks.\n"
            f"Issues: {', '.join(hard_issues)}\n"
            f"Word count: {report['word_count']} (target {report['target_words']}, range {report['min_words']}-{report['max_words']}).\n"
            f"Missing H2: {', '.join(report['missing_h2']) if report['missing_h2'] else '-'}\n"
            f"{citation_policy_note}"
            f"{personalization_note}"
            "Pertahankan fakta dan konteks. Keluarkan versi final saja.\n\n"
            f"DRAFT:\n{content}"
        )
        try:
            retry = self.ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': retry_prompt}
                ],
                options=self._chapter_generation_options(target_words, purpose="retry")
            )
            improved = (retry.get('message', {}).get('content', '') or '').strip()
            if improved:
                content = self._apply_draft_repairs(
                    chapter=chapter,
                    content=improved,
                    client=client,
                    allowed_external_citations=allowed,
                    personalization_pack=personalization_pack
                )
        except Exception:
            pass

        final_report = self._evaluate_chapter_quality(
            chapter,
            content,
            client,
            target_words=target_words,
            allowed_external_citations=allowed,
            personalization_pack=personalization_pack
        )
        if any(issue in final_report.get("issues", []) for issue in {"missing_h2", "list_structure", "citation_policy", "missing_personalization"}):
            content = self._apply_draft_repairs(
                chapter=chapter,
                content=content,
                client=client,
                allowed_external_citations=allowed,
                personalization_pack=personalization_pack
            )
            final_report = self._evaluate_chapter_quality(
                chapter,
                content,
                client,
                target_words=target_words,
                allowed_external_citations=allowed,
                personalization_pack=personalization_pack
            )

        if final_report["issues"]:
            logger.warning(f"Quality checks not fully satisfied for {chapter['title']}: {', '.join(final_report['issues'])}")
        return content

    def _tighten_chapter(
        self,
        chapter: Dict[str, Any],
        prompt: str,
        content: str,
        target_words: int,
        allowed_external_citations: Optional[Set[str]] = None
    ) -> str:
        allowed = set(allowed_external_citations or set())
        try:
            res = self.ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': (
                        f"Rapikan dan padatkan konten {chapter['title']} menjadi sekitar {target_words} kata. "
                        "Pertahankan semua heading H2 wajib, poin kunci, dan keterbacaan eksekutif. "
                        "Hapus repetisi dan kalimat pengisi.\n\n"
                        f"KONTEN SAAT INI:\n{content}"
                    )}
                ],
                options=self._chapter_generation_options(target_words, purpose="tighten")
            )
            revised = (res.get('message', {}).get('content', '') or '').strip()
            return self._clean_external_citations(revised or content, allowed)
        except Exception:
            return self._clean_external_citations(content, allowed)

    def _fit_into_word_budget(
        self,
        chapter_outputs: Dict[str, str],
        chapter_prompts: Dict[str, str],
        chapter_map: Dict[str, Dict[str, Any]],
        chapter_targets: Dict[str, int],
        max_words: int,
        allowed_external_citations: Optional[Set[str]] = None
    ) -> Dict[str, str]:
        outputs = dict(chapter_outputs)

        def total_words() -> int:
            return sum(self._word_count(text) for text in outputs.values() if text)

        current = total_words()
        if current <= max_words:
            return outputs

        for _ in range(3):
            current = total_words()
            if current <= max_words:
                break

            overflow = current - max_words
            changed = False
            chapter_order = sorted(
                [cid for cid, text in outputs.items() if text],
                key=lambda cid: (self._chapter_compression_rank(cid), -self._word_count(outputs[cid]))
            )

            for cid in chapter_order:
                if overflow <= 0:
                    break
                chapter = chapter_map.get(cid)
                if not chapter:
                    continue
                prompt = chapter_prompts.get(cid)

                current_words = self._word_count(outputs[cid])
                minimum = self._chapter_floor_words(cid, for_compression=True)
                reducible = max(0, current_words - minimum)
                if reducible < 80:
                    continue

                target = max(minimum, current_words - min(reducible, max(120, overflow)))
                target = min(target, max(minimum, int(chapter_targets.get(cid, target) * 0.95)))
                if prompt:
                    compressed = self._tighten_chapter(
                        chapter,
                        prompt,
                        outputs[cid],
                        target,
                        allowed_external_citations=allowed_external_citations
                    )
                elif self._use_structured_chapter(cid):
                    compressed = self._tighten_structured_chapter(
                        chapter=chapter,
                        content=outputs[cid],
                        target_words=target,
                    )
                else:
                    continue
                if compressed == outputs[cid]:
                    continue
                outputs[cid] = compressed
                overflow = total_words() - max_words
                changed = True

            if not changed:
                break

        if total_words() > max_words:
            ratio = max_words / max(total_words(), 1)
            ordered_items = sorted(
                list(outputs.items()),
                key=lambda item: (self._chapter_compression_rank(item[0]), -self._word_count(item[1] or ""))
            )
            for cid, text in ordered_items:
                chapter = chapter_map.get(cid)
                prompt = chapter_prompts.get(cid)
                if not chapter or not text:
                    continue
                minimum = self._chapter_floor_words(cid, for_compression=True)
                current_words = self._word_count(text)
                target = max(minimum, int(current_words * ratio))
                if target >= current_words - 60:
                    continue
                if prompt:
                    outputs[cid] = self._tighten_chapter(
                        chapter,
                        prompt,
                        text,
                        target,
                        allowed_external_citations=allowed_external_citations
                    )
                elif self._use_structured_chapter(cid):
                    outputs[cid] = self._tighten_structured_chapter(
                        chapter=chapter,
                        content=text,
                        target_words=target,
                    )

        return outputs

    def generate_document(
        self,
        client: str,
        project: str,
        budget: str,
        service_type: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        chapter_id: Optional[str] = None,
        proposal_mode: str = "canvassing"
    ) -> Tuple[Document, str, Dict[str, Any]]:
        selected_chapters = self._resolve_chapters(chapter_id, proposal_mode=proposal_mode)
        chapter_targets = self._chapter_word_targets(selected_chapters)
        content_word_budget = self._content_word_budget()

        use_demo_logic = self.firm_api.uses_demo_logic()
        app_state_store = getattr(self, "app_state_store", None)
        firm_data = self.firm_api.get_project_standards(project_type)
        firm_profile = self.firm_api.get_firm_profile()
        if app_state_store:
            firm_profile = app_state_store.enrich_firm_profile(firm_profile)
        base_client = re.sub(r'\b(Cabang|Branch|Tbk)\b.*$|^(PT\.|CV\.)', '', client, flags=re.IGNORECASE).strip()
        ai_context = " ".join([project, project_goal, project_type, service_type, proposal_mode, notes]).strip()
        relationship_context = self.firm_api.get_client_relationship(base_client)
        research_bundle = self._get_research_bundle(
            base_client,
            regulations,
            include_collaboration=use_demo_logic,
            ai_context=ai_context,
        )
        if not use_demo_logic:
            research_bundle = dict(research_bundle)
            research_bundle["collaboration"] = relationship_context.get("summary", "")
        personalization_pack = self._build_personalization_pack(
            client=client,
            project=project,
            project_goal=project_goal,
            project_type=project_type,
            timeline=timeline,
            notes=notes,
            regulations=regulations,
            research_bundle=research_bundle,
            relationship_context=relationship_context
        )
        value_map = self._build_value_map(
            client=client,
            project=project,
            service_type=service_type,
            project_goal=project_goal,
            project_type=project_type,
            timeline=timeline,
            notes=notes,
            regulations=regulations,
            firm_data=firm_data,
            firm_profile=firm_profile,
            personalization_pack=personalization_pack,
        )
        allowed_external_citations = self._collect_allowed_external_citations(research_bundle)
        proposal_contract = self._build_proposal_contract(
            client=client,
            project=project,
            budget=budget,
            service_type=service_type,
            project_goal=project_goal,
            project_type=project_type,
            timeline=timeline,
            notes=notes,
            regulations=regulations,
            selected_chapters=selected_chapters,
            research_bundle=research_bundle,
            firm_data=firm_data,
            personalization_pack=personalization_pack,
            value_map=value_map,
            proposal_mode=proposal_mode,
        )
        logo_future = self.io_pool.submit(LogoManager.get_logo_and_color, base_client)

        structured_ids = {
            chapter["id"] for chapter in selected_chapters
            if self._use_structured_chapter(chapter["id"])
        }
        context_futures = {
            chapter['id']: self.io_pool.submit(
                self._build_chapter_prompt,
                chapter, client, project, budget, service_type, project_goal, project_type, timeline,
                notes, regulations, firm_data, firm_profile, research_bundle, personalization_pack, value_map, proposal_contract, proposal_mode,
                chapter_targets.get(chapter['id'], self._target_words(chapter))
            )
            for chapter in selected_chapters
            if chapter['id'] not in structured_ids
        }

        chapter_map = {chapter['id']: chapter for chapter in selected_chapters}
        chapter_prompts: Dict[str, str] = {}
        chapter_outputs: Dict[str, str] = {}
        for chapter in selected_chapters:
            if chapter['id'] in structured_ids:
                chapter_outputs[chapter['id']] = self._clean_external_citations(
                    self._render_structured_chapter(
                        chapter=chapter,
                        client=client,
                        project=project,
                        budget=budget,
                        service_type=service_type,
                        project_goal=project_goal,
                        project_type=project_type,
                        timeline=timeline,
                        notes=notes,
                        regulations=regulations,
                        firm_data=firm_data,
                        firm_profile=firm_profile,
                        personalization_pack=personalization_pack,
                        value_map=value_map,
                        proposal_mode=proposal_mode,
                    ),
                    allowed_external_citations
                )
                continue
            ctx = context_futures[chapter['id']].result()
            if not ctx['success']:
                continue
            chapter_prompts[chapter['id']] = ctx['prompt']
            try:
                chapter_outputs[chapter['id']] = self._draft_chapter(
                    chapter=chapter,
                    prompt=ctx['prompt'],
                    client=client,
                    target_words=chapter_targets.get(chapter['id'], self._target_words(chapter)),
                    allowed_external_citations=allowed_external_citations,
                    personalization_pack=personalization_pack
                )
            except Exception as e:
                logger.error(f"Generation Error for {chapter['title']}: {e}")

        chapter_outputs = {
            chapter_id: self._clean_external_citations(content, allowed_external_citations)
            for chapter_id, content in chapter_outputs.items()
        }

        chapter_outputs = self._fit_into_word_budget(
            chapter_outputs=chapter_outputs,
            chapter_prompts=chapter_prompts,
            chapter_map=chapter_map,
            chapter_targets=chapter_targets,
            max_words=content_word_budget,
            allowed_external_citations=allowed_external_citations
        )
        chapter_outputs = self._stabilize_chapter_outputs(
            chapter_outputs=chapter_outputs,
            selected_chapters=selected_chapters,
            chapter_targets=chapter_targets,
            client=client,
            timeline=timeline,
            allowed_external_citations=allowed_external_citations,
            personalization_pack=personalization_pack,
        )
        if not self._throughput_mode():
            chapter_outputs = self._apply_global_coherence(chapter_outputs, selected_chapters, client, project)
            chapter_outputs = {
                chapter_id: self._clean_external_citations(content, allowed_external_citations)
                for chapter_id, content in chapter_outputs.items()
            }
            chapter_outputs = self._fit_into_word_budget(
                chapter_outputs=chapter_outputs,
                chapter_prompts=chapter_prompts,
                chapter_map=chapter_map,
                chapter_targets=chapter_targets,
                max_words=content_word_budget,
                allowed_external_citations=allowed_external_citations
            )
            chapter_outputs = self._stabilize_chapter_outputs(
                chapter_outputs=chapter_outputs,
                selected_chapters=selected_chapters,
                chapter_targets=chapter_targets,
                client=client,
                timeline=timeline,
                allowed_external_citations=allowed_external_citations,
                personalization_pack=personalization_pack,
            )
        generated_words = sum(self._word_count(t) for t in chapter_outputs.values() if t)
        estimated_pages = self._estimated_pages(generated_words)
        if estimated_pages > MAX_PROPOSAL_PAGES:
            logger.warning(
                f"Estimated page count is above limit ({estimated_pages}>{MAX_PROPOSAL_PAGES}); applying one more compacting pass."
            )
            tighter_budget = max(int(content_word_budget * 0.94), 4000)
            chapter_outputs = self._fit_into_word_budget(
                chapter_outputs=chapter_outputs,
                chapter_prompts=chapter_prompts,
                chapter_map=chapter_map,
                chapter_targets=chapter_targets,
                max_words=tighter_budget,
                allowed_external_citations=allowed_external_citations
            )

        if chapter_outputs.get("c_closing"):
            chapter_outputs["c_closing"] = self._inject_verified_firm_contact(
                chapter_outputs["c_closing"],
                firm_profile
            )
            # Enhance closing with comprehensive firm information via OSINT
            try:
                chapter_outputs["c_closing"] = self._enhance_closing_with_firm_osint(
                    chapter_outputs["c_closing"],
                    firm_profile
                )
            except Exception as e:
                logger.warning(f"Could not enhance closing with OSINT firm information: {e}")

        acceptance_report = self._evaluate_proposal_acceptance(
            chapter_outputs=chapter_outputs,
            selected_chapters=selected_chapters,
            chapter_targets=chapter_targets,
            client=client,
            project=project,
            notes=notes,
            firm_profile=firm_profile,
            allowed_external_citations=allowed_external_citations,
            personalization_pack=personalization_pack,
            value_map=value_map,
        )
        if not acceptance_report["passes"]:
            weak_chapters = self._select_improvement_chapters(acceptance_report, selected_chapters)
            for chapter_id in weak_chapters:
                chapter = chapter_map.get(chapter_id)
                prompt = chapter_prompts.get(chapter_id)
                content = chapter_outputs.get(chapter_id, "")
                if not chapter or not prompt or not content:
                    continue
                chapter_outputs[chapter_id] = self._improve_weak_chapter(
                    chapter=chapter,
                    prompt=prompt,
                    content=content,
                    client=client,
                    target_words=chapter_targets.get(chapter_id, self._target_words(chapter)),
                    acceptance_report=acceptance_report,
                    personalization_pack=personalization_pack,
                    value_map=value_map,
                    allowed_external_citations=allowed_external_citations,
                    timeline=timeline,
                )

            if weak_chapters:
                chapter_outputs = {
                    chapter_id: self._clean_external_citations(content, allowed_external_citations)
                    for chapter_id, content in chapter_outputs.items()
                }
                chapter_outputs = self._fit_into_word_budget(
                    chapter_outputs=chapter_outputs,
                    chapter_prompts=chapter_prompts,
                    chapter_map=chapter_map,
                    chapter_targets=chapter_targets,
                    max_words=content_word_budget,
                    allowed_external_citations=allowed_external_citations
                )
                chapter_outputs = self._stabilize_chapter_outputs(
                    chapter_outputs=chapter_outputs,
                    selected_chapters=selected_chapters,
                    chapter_targets=chapter_targets,
                    client=client,
                    timeline=timeline,
                    allowed_external_citations=allowed_external_citations,
                    personalization_pack=personalization_pack,
                )
                if chapter_outputs.get("c_closing"):
                    chapter_outputs["c_closing"] = self._inject_verified_firm_contact(
                        chapter_outputs["c_closing"],
                        firm_profile
                    )
                    # Enhance closing with comprehensive firm information via OSINT
                    try:
                        chapter_outputs["c_closing"] = self._enhance_closing_with_firm_osint(
                            chapter_outputs["c_closing"],
                            firm_profile
                        )
                    except Exception as e:
                        logger.warning(f"Could not enhance closing with OSINT firm information: {e}")
                acceptance_report = self._evaluate_proposal_acceptance(
                    chapter_outputs=chapter_outputs,
                    selected_chapters=selected_chapters,
                    chapter_targets=chapter_targets,
                    client=client,
                    project=project,
                    notes=notes,
                    firm_profile=firm_profile,
                    allowed_external_citations=allowed_external_citations,
                    personalization_pack=personalization_pack,
                    value_map=value_map,
                )

        if not acceptance_report["passes"]:
            logger.warning(
                "Proposal acceptance below target | score=%s | categories=%s | ai_adoption_fit=%s | hard_failures=%s | low_categories=%s | soft_findings=%s",
                acceptance_report["score"],
                acceptance_report["categories"],
                acceptance_report.get("ai_adoption_fit"),
                acceptance_report["hard_failures"],
                acceptance_report["low_categories"],
                acceptance_report.get("soft_findings"),
            )
        else:
            logger.info(
                "Proposal acceptance passed | score=%s | categories=%s | ai_adoption_fit=%s | estimated_pages=%s",
                acceptance_report["score"],
                acceptance_report["categories"],
                acceptance_report.get("ai_adoption_fit"),
                acceptance_report["estimated_pages"],
            )

        try:
            logo_stream, theme_color = logo_future.result(timeout=8)
        except Exception:
            logo_stream, theme_color = None, DEFAULT_COLOR

        template_path = app_state_store.get_template_path() if app_state_store else ""
        doc, using_template = DocumentBuilder.create_base_document(template_path)
        StyleEngine.apply_document_styles(doc, preserve_existing=using_template)

        DocumentBuilder.add_reference_cover_page(
            doc=doc,
            client=client,
            project=project,
            service_type=service_type,
            project_type=project_type,
            timeline=timeline,
            budget=budget,
            firm_profile=firm_profile,
            theme_color=theme_color,
            logo_stream=logo_stream,
        )

        rendered_any = False
        rendered_firm_profile = False
        for i, chapter in enumerate(selected_chapters):
            content = chapter_outputs.get(chapter['id'], '').strip()
            if not content:
                continue
            rendered_any = True
            if chapter['id'] == "c_closing":
                # The rendered DOCX appends a structured firm profile/contact block,
                # so keep the closing narrative clean by stripping the inline contact section.
                content = re.split(
                    r'(?im)^\s*##\s*Informasi Kontak dan Langkah Lanjutan\s*$',
                    content,
                    maxsplit=1,
                )[0].rstrip()
            DocumentBuilder.add_reference_chapter_heading(doc, chapter['title'], theme_color)
            DocumentBuilder.process_content(doc, content, theme_color, chapter['title'])
            if chapter['id'] == "c_closing" and not rendered_firm_profile:
                DocumentBuilder.add_writer_firm_profile_section(doc, firm_profile, theme_color)
                rendered_firm_profile = True

            has_next = any(
                chapter_outputs.get(next_chapter['id'], '').strip()
                for next_chapter in selected_chapters[i + 1:]
            )
            if has_next:
                doc.add_page_break()

        if not rendered_any:
            doc.add_paragraph("Konten proposal belum berhasil digenerate. Mohon ulangi proses.")
        DocumentBuilder.compact_layout(doc)

        base_name = f"Proposal_{client}_{project}"
        if len(selected_chapters) == 1:
            chapter_slug = re.sub(r'[^A-Za-z0-9]+', '_', selected_chapters[0]['title']).strip('_')
            base_name = f"{base_name}_{chapter_slug}"

        metadata = {
            "acceptance_report": acceptance_report,
            "estimated_pages": acceptance_report.get("estimated_pages"),
            "generated_words": generated_words,
            "using_template": using_template,
            "proposal_mode": proposal_mode,
        }
        return doc, base_name.replace(" ", "_"), metadata


    def run(
        self,
        client: str,
        project: str,
        budget: str,
        service_type: str,
        project_goal: str,
        project_type: str,
        timeline: str,
        notes: str,
        regulations: str,
        chapter_id: Optional[str] = None,
        proposal_mode: str = "canvassing"
    ) -> Tuple[Document, str, Dict[str, Any]]:
        return self.generate_document(
            client=client,
            project=project,
            budget=budget,
            service_type=service_type,
            project_goal=project_goal,
            project_type=project_type,
            timeline=timeline,
            notes=notes,
            regulations=regulations,
            chapter_id=chapter_id,
            proposal_mode=proposal_mode,
        )
