"""YAML Facts Indexer.

Converts structured YAML fact files into searchable document chunks.
Each fact entry becomes a standalone chunk with rich metadata for retrieval.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


def infer_entity_type_for_yaml_chunk(metadata: dict) -> str:
    """Infer high-level entity type for YAML fact chunks."""
    fact_type = str(metadata.get("fact_type", "")).strip().lower()
    topic = str(metadata.get("topic", "")).strip().lower()
    source = str(metadata.get("source", "")).strip().lower()

    if fact_type in {"deadline", "requirement", "language_requirement", "quota", "key_dates"}:
        return "admission"
    if fact_type in {"tuition", "tuition_tariff", "tuition_exemption"}:
        return "finance"
    if fact_type in {"vwu_location", "gotcha", "certificate"} or "language" in topic:
        return "language"
    if fact_type in {"contact", "general_info"} and "universities/" in source:
        return "university"
    return "general"


def _annotate_entity_type(chunks: list[dict]) -> list[dict]:
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        if "entity_type" not in metadata:
            metadata["entity_type"] = infer_entity_type_for_yaml_chunk(metadata)
    return chunks


def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten a nested dictionary with dot notation keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _format_value(value) -> str:
    """Format a value for human-readable output."""
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return ", ".join(value)
        return "\n".join(f"  - {_format_value(item)}" for item in value)
    elif isinstance(value, dict):
        lines = []
        for k, v in value.items():
            lines.append(f"  {k}: {_format_value(v)}")
        return "\n".join(lines)
    elif isinstance(value, bool):
        return "Да" if value else "Нет"
    else:
        return str(value)


def _extract_deadline_chunks(data: dict, uni_id: str, uni_name: str) -> list[dict]:
    """Extract deadline-specific chunks from university YAML."""
    chunks = []
    deadlines = data.get('deadlines', [])

    for deadline in deadlines:
        if not isinstance(deadline, dict):
            continue

        semester = deadline.get('semester', 'Unknown')
        category = deadline.get('category', 'all')
        levels = deadline.get('level', [])
        if isinstance(levels, str):
            levels = [levels]

        # Build content
        lines = [f"[Дедлайн: {uni_name} ({uni_id})]"]
        lines.append(f"Семестр: {semester}")
        lines.append(f"Категория: {category}")
        if levels:
            lines.append(f"Уровни: {', '.join(levels)}")

        submission = deadline.get('submission')
        if isinstance(submission, dict):
            lines.append(f"Подача: {submission.get('start', '?')} — {submission.get('end', '?')}")
        elif 'submission_deadline' in deadline:
            lines.append(f"Дедлайн подачи: {deadline['submission_deadline']}")

        enrollment = deadline.get('enrollment')
        if isinstance(enrollment, dict):
            lines.append(f"Зачисление: {enrollment.get('start', '?')} — {enrollment.get('end', '?')}")

        notes = deadline.get('notes', '')
        if notes:
            lines.append(f"Примечание: {notes}")

        source_url = deadline.get('source_url', data.get('source_urls', [''])[0] if isinstance(data.get('source_urls'), list) else '')

        chunks.append({
            'content': "\n".join(lines),
            'metadata': {
                'source': f"facts/universities/{uni_id}.yaml",
                'title': f"Дедлайн {uni_name}",
                'fact_type': 'deadline',
                'university': uni_id,
                'semester': semester,
                'category': category,
                'source_url': source_url,
                'is_yaml_fact': True,
            }
        })

    return chunks


def _extract_tuition_chunks(data: dict, uni_id: str, uni_name: str) -> list[dict]:
    """Extract tuition fee chunks from university YAML."""
    chunks = []
    tuition_list = data.get('tuition', [])

    for tuition in tuition_list:
        if not isinstance(tuition, dict):
            continue

        category = tuition.get('category', 'unknown')
        amount = tuition.get('amount', 0)
        currency = tuition.get('currency', 'EUR')
        period = tuition.get('period', 'semester')
        total = tuition.get('total', amount)

        lines = [f"[Стоимость обучения: {uni_name} ({uni_id})]"]
        lines.append(f"Категория: {category}")
        lines.append(f"Studienbeitrag: {amount} {currency}/{period}")

        additional = tuition.get('additional', [])
        for add in additional:
            if isinstance(add, dict):
                lines.append(f"  + {add.get('name', '?')}: {add.get('amount', '?')} {currency}")

        lines.append(f"Итого: {total} {currency}/{period}")

        notes = tuition.get('notes', '')
        if notes:
            lines.append(f"Примечание: {notes}")

        source_url = tuition.get('source_url', '')

        chunks.append({
            'content': "\n".join(lines),
            'metadata': {
                'source': f"facts/universities/{uni_id}.yaml",
                'title': f"Стоимость {uni_name}",
                'fact_type': 'tuition',
                'university': uni_id,
                'category': category,
                'source_url': source_url,
                'is_yaml_fact': True,
            }
        })

    return chunks


def _extract_language_chunks(data: dict, uni_id: str, uni_name: str) -> list[dict]:
    """Extract language requirement chunks from university YAML."""
    chunks = []
    lang_req = data.get('language_requirements', {})

    if not lang_req:
        return chunks

    german = lang_req.get('german', {})
    if german:
        lines = [f"[Требования к немецкому: {uni_name} ({uni_id})]"]

        for_reg = german.get('for_registration') or german.get('submission_minimum')
        for_enroll = german.get('for_enrollment') or german.get('enrollment')
        direct = german.get('direct_admission')

        if for_reg:
            lines.append(f"Для подачи: {for_reg}")
        if for_enroll:
            lines.append(f"Для зачисления: {for_enroll}")
        if direct:
            lines.append(f"Прямое зачисление: {direct}")

        certificates = german.get('certificates', [])
        if isinstance(certificates, list) and certificates:
            lines.append(f"Сертификаты: {', '.join(str(c) for c in certificates)}")
        elif isinstance(certificates, dict):
            for level, certs in certificates.items():
                if certs:
                    lines.append(f"  {level}: {', '.join(str(c) for c in certs) if isinstance(certs, list) else certs}")

        validity = german.get('certificate_validity_years')
        if validity:
            lines.append(f"Срок действия сертификата: {validity} года/лет")

        notes = german.get('notes', '')
        if notes:
            lines.append(f"Примечание: {notes}")

        source_url = german.get('source_url', '')

        chunks.append({
            'content': "\n".join(lines),
            'metadata': {
                'source': f"facts/universities/{uni_id}.yaml",
                'title': f"Немецкий для {uni_name}",
                'fact_type': 'language_requirement',
                'university': uni_id,
                'language': 'german',
                'source_url': source_url,
                'is_yaml_fact': True,
            }
        })

    english = lang_req.get('english', {})
    if english:
        lines = [f"[Требования к английскому: {uni_name} ({uni_id})]"]

        required_for = english.get('required_for', [])
        if required_for:
            lines.append(f"Требуется для: {', '.join(required_for) if isinstance(required_for, list) else required_for}")

        min_level = english.get('min_level')
        if min_level:
            lines.append(f"Минимальный уровень: {min_level}")

        certificates = english.get('certificates', [])
        if certificates:
            for cert in certificates:
                if isinstance(cert, dict):
                    cert_name = cert.get('name', '?')
                    min_score = cert.get('min_score', '')
                    lines.append(f"  - {cert_name}: {min_score}")
                else:
                    lines.append(f"  - {cert}")

        notes = english.get('notes', '')
        if notes:
            lines.append(f"Примечание: {notes}")

        source_url = english.get('source_url', '')

        chunks.append({
            'content': "\n".join(lines),
            'metadata': {
                'source': f"facts/universities/{uni_id}.yaml",
                'title': f"Английский для {uni_name}",
                'fact_type': 'language_requirement',
                'university': uni_id,
                'language': 'english',
                'source_url': source_url,
                'is_yaml_fact': True,
            }
        })

    return chunks


def _extract_quota_chunks(data: dict, uni_id: str, uni_name: str) -> list[dict]:
    """Extract quota/places chunks (especially for MedUni)."""
    chunks = []
    quotas = data.get('quotas', {})

    if not quotas:
        return chunks

    for program_key, program_data in quotas.items():
        if not isinstance(program_data, dict):
            continue

        # Skip non-program keys
        if program_key in ('source_url', 'last_verified', 'distribution_process', 'quota_determination'):
            continue

        lines = [f"[Квоты/Места: {uni_name} - {program_key}]"]

        total = program_data.get('total_places')
        if total:
            lines.append(f"Всего мест: {total}")

        for quota_key, quota_data in program_data.items():
            if quota_key == 'total_places':
                continue
            if isinstance(quota_data, dict):
                percentage = quota_data.get('percentage', '?')
                places = quota_data.get('places', '?')
                condition = quota_data.get('condition', '')
                lines.append(f"{quota_key}: {percentage}% ({places} мест)")
                if condition:
                    lines.append(f"  Условие: {condition}")
            elif quota_key.endswith('_places'):
                lines.append(f"{quota_key}: {quota_data}")

        source_url = quotas.get('source_url', '')

        chunks.append({
            'content': "\n".join(lines),
            'metadata': {
                'source': f"facts/universities/{uni_id}.yaml",
                'title': f"Квоты {uni_name}",
                'fact_type': 'quota',
                'university': uni_id,
                'program': program_key,
                'source_url': source_url,
                'is_yaml_fact': True,
            }
        })

    return chunks


def _extract_contact_chunks(data: dict, uni_id: str, uni_name: str) -> list[dict]:
    """Extract contact information chunks."""
    chunks = []
    contacts = data.get('contacts', {})

    if not contacts:
        return chunks

    for dept_key, dept_data in contacts.items():
        if not isinstance(dept_data, dict):
            continue

        lines = [f"[Контакты: {uni_name} - {dept_key}]"]

        name = dept_data.get('name', dept_key)
        lines.append(f"Отдел: {name}")

        address = dept_data.get('address')
        if address:
            lines.append(f"Адрес: {address}")

        email = dept_data.get('email')
        if email:
            lines.append(f"Email: {email}")

        phone = dept_data.get('phone')
        if phone:
            lines.append(f"Телефон: {phone}")

        website = dept_data.get('website')
        if website:
            lines.append(f"Сайт: {website}")

        hours = dept_data.get('hours')
        if hours:
            lines.append(f"Часы работы: {hours}")

        chunks.append({
            'content': "\n".join(lines),
            'metadata': {
                'source': f"facts/universities/{uni_id}.yaml",
                'title': f"Контакты {uni_name}",
                'fact_type': 'contact',
                'university': uni_id,
                'department': dept_key,
                'is_yaml_fact': True,
            }
        })

    return chunks


def _extract_key_dates_chunks(data: dict, uni_id: str, uni_name: str) -> list[dict]:
    """Extract key dates chunks (MedAT, entrance exams, etc.)."""
    chunks = []

    # MedAT key dates
    key_dates = data.get('key_dates_2026', {})
    if key_dates:
        lines = [f"[Ключевые даты 2026: {uni_name}]"]

        reg_open = key_dates.get('registration_open')
        if reg_open:
            lines.append(f"Открытие регистрации: {reg_open}")

        reg_deadline = key_dates.get('registration_deadline')
        if reg_deadline:
            lines.append(f"Дедлайн регистрации: {reg_deadline}")

        reg_fee = key_dates.get('registration_fee')
        if reg_fee:
            lines.append(f"Стоимость регистрации: €{reg_fee}")

        medat_date = key_dates.get('medat_date')
        if medat_date:
            lines.append(f"Дата MedAT: {medat_date}")

        results = key_dates.get('results')
        if results:
            lines.append(f"Результаты: {results}")

        critical_notes = key_dates.get('critical_notes', [])
        for note in critical_notes:
            lines.append(f"⚠️ {note}")

        source_url = key_dates.get('source_url', '')

        chunks.append({
            'content': "\n".join(lines),
            'metadata': {
                'source': f"facts/universities/{uni_id}.yaml",
                'title': f"Ключевые даты {uni_name} 2026",
                'fact_type': 'key_dates',
                'university': uni_id,
                'year': 2026,
                'source_url': source_url,
                'is_yaml_fact': True,
            }
        })

    return chunks


def parse_university_yaml(file_path: str) -> list[dict]:
    """Parse a university YAML file and extract all fact chunks.

    Args:
        file_path: Path to the university YAML file

    Returns:
        List of chunk dicts with 'content' and 'metadata'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not data:
        return []

    uni_id = data.get('id', Path(file_path).stem)
    names = data.get('name', {})
    uni_name = names.get('de') or names.get('en') or names.get('ru') or uni_id

    all_chunks = []

    # Extract different fact types
    all_chunks.extend(_extract_deadline_chunks(data, uni_id, uni_name))
    all_chunks.extend(_extract_tuition_chunks(data, uni_id, uni_name))
    all_chunks.extend(_extract_language_chunks(data, uni_id, uni_name))
    all_chunks.extend(_extract_quota_chunks(data, uni_id, uni_name))
    all_chunks.extend(_extract_contact_chunks(data, uni_id, uni_name))
    all_chunks.extend(_extract_key_dates_chunks(data, uni_id, uni_name))

    # Also create a general info chunk
    lines = [f"[Информация: {uni_name} ({uni_id})]"]
    city = data.get('city')
    if city:
        lines.append(f"Город: {city}")
    website = data.get('website')
    if website:
        lines.append(f"Сайт: {website}")
    students = data.get('students_count')
    if students:
        lines.append(f"Студентов: {students}")
    founded = data.get('founded')
    if founded:
        lines.append(f"Основан: {founded}")
    description = data.get('description')
    if description:
        lines.append(f"Описание: {description}")

    all_chunks.append({
        'content': "\n".join(lines),
        'metadata': {
            'source': f"facts/universities/{uni_id}.yaml",
            'title': uni_name,
            'fact_type': 'general_info',
            'university': uni_id,
            'source_url': website or '',
            'is_yaml_fact': True,
        }
    })

    return all_chunks


def parse_language_yaml(file_path: str) -> list[dict]:
    """Parse a language-related YAML file and extract fact chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not data:
        return []

    file_id = data.get('id', Path(file_path).stem)
    chunks = []

    # For german-requirements.yaml
    if 'gotchas' in data:
        for gotcha_id, gotcha in data.get('gotchas', {}).items():
            if isinstance(gotcha, dict):
                lines = [f"[Важно: {gotcha.get('title', gotcha_id)}]"]
                lines.append(f"Уровень важности: {gotcha.get('severity', 'high')}")
                desc = gotcha.get('description', '')
                if desc:
                    lines.append(desc)
                rec = gotcha.get('recommendation')
                if rec:
                    lines.append(f"Рекомендация: {rec}")

                chunks.append({
                    'content': "\n".join(lines),
                    'metadata': {
                        'source': f"facts/language/{Path(file_path).name}",
                        'title': gotcha.get('title', gotcha_id),
                        'fact_type': 'gotcha',
                        'topic': 'german_requirements',
                        'is_yaml_fact': True,
                    }
                })

    # For certificate info
    if 'certificates' in data:
        for level, certs in data.get('certificates', {}).items():
            if isinstance(certs, list):
                for cert in certs:
                    if isinstance(cert, dict):
                        lines = [f"[Сертификат: {cert.get('name', '?')} (уровень {level})]"]
                        org = cert.get('organization')
                        if org:
                            lines.append(f"Организация: {org}")
                        cost = cert.get('cost_range') or cert.get('cost')
                        if cost:
                            lines.append(f"Стоимость: {cost}")
                        from_russia = cert.get('from_russia')
                        if from_russia:
                            lines.append(f"Из России: {from_russia}")
                        notes = cert.get('notes')
                        if notes:
                            lines.append(f"Примечание: {notes}")

                        chunks.append({
                            'content': "\n".join(lines),
                            'metadata': {
                                'source': f"facts/language/{Path(file_path).name}",
                                'title': f"Сертификат {cert.get('name', '?')}",
                                'fact_type': 'certificate',
                                'level': level,
                                'is_yaml_fact': True,
                            }
                        })

    # For VWU locations
    if 'locations' in data:
        for loc_id, loc_data in data.get('locations', {}).items():
            if isinstance(loc_data, dict):
                lines = [f"[VWU: {loc_data.get('name', loc_id)}]"]
                full_name = loc_data.get('full_name')
                if full_name:
                    lines.append(f"Полное название: {full_name}")

                cost = loc_data.get('cost_per_semester')
                if cost:
                    lines.append(f"Стоимость: €{cost}/семестр")

                cost_range = loc_data.get('cost_range')
                if cost_range:
                    lines.append(f"Стоимость: {cost_range}")

                cost_options = loc_data.get('cost_options', [])
                for opt in cost_options:
                    if isinstance(opt, dict):
                        lines.append(f"  {opt.get('hours', '?')}: €{opt.get('cost', '?')}")

                notes = loc_data.get('notes')
                if notes:
                    lines.append(f"Примечание: {notes}")

                warning = loc_data.get('warning')
                if warning:
                    lines.append(f"⚠️ {warning}")

                chunks.append({
                    'content': "\n".join(lines),
                    'metadata': {
                        'source': f"facts/language/{Path(file_path).name}",
                        'title': loc_data.get('name', loc_id),
                        'fact_type': 'vwu_location',
                        'topic': 'vorstudienlehrgang',
                        'is_yaml_fact': True,
                    }
                })

    return chunks


def parse_financial_yaml(file_path: str) -> list[dict]:
    """Parse a financial YAML file and extract fact chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not data:
        return []

    chunks = []

    # Tariffs
    for tariff_id, tariff_data in data.get('tariffs', {}).items():
        if isinstance(tariff_data, dict):
            lines = [f"[Тариф: {tariff_id}]"]
            studienbeitrag = tariff_data.get('studienbeitrag', 0)
            oeh = tariff_data.get('oeh_beitrag', 25.20)
            total = tariff_data.get('total', studienbeitrag + oeh)
            currency = tariff_data.get('currency', 'EUR')

            lines.append(f"Studienbeitrag: €{studienbeitrag}")
            lines.append(f"ÖH-Beitrag: €{oeh}")
            lines.append(f"Итого: €{total}/семестр")

            notes = tariff_data.get('notes')
            if notes:
                lines.append(f"Примечание: {notes}")

            chunks.append({
                'content': "\n".join(lines),
                'metadata': {
                    'source': f"facts/financial/{Path(file_path).name}",
                    'title': f"Тариф {tariff_id}",
                    'fact_type': 'tuition_tariff',
                    'category': tariff_id,
                    'is_yaml_fact': True,
                }
            })

    # Exemptions
    for exemption_id, exemption_data in data.get('exemptions', {}).items():
        if isinstance(exemption_data, dict):
            lines = [f"[Освобождение: {exemption_id}]"]
            desc = exemption_data.get('description', '')
            if desc:
                lines.append(desc)

            condition = exemption_data.get('condition')
            if condition:
                lines.append(f"Условие: {condition}")

            russia = exemption_data.get('russia_eligible')
            if russia is not None:
                lines.append(f"Для граждан России: {'Да' if russia else 'Нет'}")

            chunks.append({
                'content': "\n".join(lines),
                'metadata': {
                    'source': f"facts/financial/{Path(file_path).name}",
                    'title': f"Освобождение {exemption_id}",
                    'fact_type': 'tuition_exemption',
                    'is_yaml_fact': True,
                }
            })

    return chunks


def index_all_yaml_facts(facts_dir: str = "facts") -> list[dict]:
    """Index all YAML fact files from the facts directory.

    Args:
        facts_dir: Path to the facts directory

    Returns:
        List of all fact chunks from all YAML files
    """
    all_chunks = []
    facts_path = Path(facts_dir)

    if not facts_path.exists():
        logger.warning(f"Facts directory not found: {facts_dir}")
        return []

    # Universities
    unis_path = facts_path / "universities"
    if unis_path.exists():
        for yaml_file in unis_path.glob("*.yaml"):
            try:
                chunks = parse_university_yaml(str(yaml_file))
                all_chunks.extend(chunks)
                logger.info(f"  📊 {yaml_file.name}: {len(chunks)} fact chunks")
            except Exception as e:
                logger.error(f"Error parsing {yaml_file}: {e}")

    # Language
    lang_path = facts_path / "language"
    if lang_path.exists():
        for yaml_file in lang_path.glob("*.yaml"):
            try:
                chunks = parse_language_yaml(str(yaml_file))
                all_chunks.extend(chunks)
                logger.info(f"  📊 {yaml_file.name}: {len(chunks)} fact chunks")
            except Exception as e:
                logger.error(f"Error parsing {yaml_file}: {e}")

    # Financial
    fin_path = facts_path / "financial"
    if fin_path.exists():
        for yaml_file in fin_path.glob("*.yaml"):
            try:
                chunks = parse_financial_yaml(str(yaml_file))
                all_chunks.extend(chunks)
                logger.info(f"  📊 {yaml_file.name}: {len(chunks)} fact chunks")
            except Exception as e:
                logger.error(f"Error parsing {yaml_file}: {e}")

    _annotate_entity_type(all_chunks)
    logger.info(f"Total YAML fact chunks: {len(all_chunks)}")
    return all_chunks
