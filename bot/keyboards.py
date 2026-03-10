from telegram import InlineKeyboardButton, InlineKeyboardMarkup

def country_keyboard():
    keyboard = [
        [InlineKeyboardButton("🇺🇦 Украина", callback_data="country_UA")],
        [InlineKeyboardButton("🇷🇺 Россия / 🇧🇾 Беларусь", callback_data="country_RU_BY")],
        [InlineKeyboardButton("🇰🇿 Казахстан / СНГ", callback_data="country_CIS")],
        [InlineKeyboardButton("🌍 Другое", callback_data="country_ALL")]
    ]
    return InlineKeyboardMarkup(keyboard)

def document_keyboard():
    keyboard = [
        [InlineKeyboardButton("📄 Школьный аттестат", callback_data="doc_school")],
        [InlineKeyboardButton("🎓 Диплом бакалавра", callback_data="doc_bachelor")],
        [InlineKeyboardButton("🎓 Диплом магистра", callback_data="doc_master")]
    ]
    return InlineKeyboardMarkup(keyboard)

def target_level_keyboard():
    keyboard = [
        [InlineKeyboardButton("🎯 Бакалавриат", callback_data="target_bachelor")],
        [InlineKeyboardButton("🎯 Магистратура", callback_data="target_master")]
    ]
    return InlineKeyboardMarkup(keyboard)

def german_level_keyboard():
    keyboard = [
        [InlineKeyboardButton("🔴 Пока ноль", callback_data="german_zero")],
        [InlineKeyboardButton("🟠 A2", callback_data="german_A2")],
        [InlineKeyboardButton("🟡 B1", callback_data="german_B1")],
        [InlineKeyboardButton("🟢 B2 (достаточно для большинства вузов)", callback_data="german_B2")],
        [InlineKeyboardButton("🔵 C1 (достаточно для всех вузов)", callback_data="german_C1")],
    ]
    return InlineKeyboardMarkup(keyboard)

def english_level_keyboard():
    keyboard = [
        [InlineKeyboardButton("🔴 Нет / ниже B1", callback_data="english_none")],
        [InlineKeyboardButton("🟡 B1", callback_data="english_B1")],
        [InlineKeyboardButton("🟢 B2 (мин. для EN-программ)", callback_data="english_B2")],
        [InlineKeyboardButton("🔵 C1+ (достаточно для всех EN-программ)", callback_data="english_C1")],
    ]
    return InlineKeyboardMarkup(keyboard)


def suggested_questions_keyboard(questions: list, msg_id: int = None) -> InlineKeyboardMarkup:
    """Create an inline keyboard from a list of suggested follow-up questions."""
    keyboard = []
    for i, q in enumerate(questions[:5]):  # max 5 buttons
        label = q if len(q) <= 60 else q[:57] + "..."
        # If msg_id is provided, use sq_{msg_id}_{idx} format for better tracking.
        # Otherwise, fall back to legacy suggest_{idx}.
        callback_data = f"sq_{msg_id}_{i}" if msg_id else f"suggest_{i}"
        keyboard.append([InlineKeyboardButton(f"💬 {label}", callback_data=callback_data)])
    return InlineKeyboardMarkup(keyboard)
