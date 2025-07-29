# Создание и настройка графа конвейера
from langgraph.graph import StateGraph

import nltk
nltk.download('punkt')

from app.letter_pipeline.nodes import input_node, search_chunks_node, build_prompt_node, generate_letter_node, \
    output_node
from app.letter_pipeline.types import LetterState
graph = StateGraph(LetterState)

"""
Граф для обработки конвейера генерации письма.

Состоит из узлов: input, search, prompt, generate, output.
Каждый узел обновляет состояние LetterState, добавляя данные или возвращая итоговое письмо.
"""

# Добавление узлов в граф
graph.add_node("input", input_node)
graph.add_node("search", search_chunks_node)
graph.add_node("prompt", build_prompt_node)
graph.add_node("generate", generate_letter_node)
graph.add_node("output", output_node)

# Установка точки входа
graph.set_entry_point("input")

# Добавление связей между узлами
graph.add_edge("input", "search")
graph.add_edge("search", "prompt")
graph.add_edge("prompt", "generate")
graph.add_edge("generate", "output")

# Установка точки выхода
graph.set_finish_point("output")

# Компиляция графа в исполняемый конвейер
chain = graph.compile()
"""
Скомпилированный конвейер для генерации письма.

Обрабатывает пользовательский ввод, выполняет поиск чанков, формирует промпт,
генерирует письмо и возвращает результат.
"""
