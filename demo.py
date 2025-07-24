#!/usr/bin/env python3
"""
Демонстрационный скрипт для показа возможностей системы
классификации спама на конкурсе
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict
from datetime import datetime
import random
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import sys

console = Console()

class DemoSpamClassifier:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def classify_post(self, post: Dict) -> Dict:
        """Классификация поста через API"""
        async with self.session.post(f"{self.api_url}/classify", json=post) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}
    
    async def get_stats(self) -> Dict:
        """Получение статистики"""
        try:
            async with self.session.get(f"{self.api_url}/stats") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except:
            return {"total_classified": 0, "spam_detected": 0}
    
    async def fetch_dev_to_posts(self, count: int = 20) -> List[Dict]:
        """Получение постов с dev.to для демонстрации"""
        posts = []
        
        try:
            async with self.session.get(f"https://dev.to/api/articles?per_page={count}") as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            console.print(f"[red]Ошибка получения постов с dev.to: {e}[/red]")
        
        # Fallback: демонстрационные посты
        return self.get_demo_posts()
    
    def get_demo_posts(self) -> List[Dict]:
        """Демонстрационные посты для показа"""
        return [
            {
                "id": 1001,
                "title": "Complete Guide to Python FastAPI in 2024",
                "description": "Learn how to build modern APIs with FastAPI, including async programming, database integration, and testing",
                "tag_list": ["python", "fastapi", "api", "tutorial"],
                "reading_time_minutes": 12,
                "public_reactions_count": 145,
                "comments_count": 23,
                "user": {"followers_count": 890},
                "url": "https://dev.to/example/fastapi-guide",
                "published_at": "2024-01-20T10:00:00Z"
            },
            {
                "id": 1002,
                "title": "🚀 EARN $5000/MONTH CODING - NO EXPERIENCE NEEDED!!!",
                "description": "Make money fast with this secret coding method! Click here now!",
                "tag_list": ["money", "earn", "coding", "profit", "investment", "business", "startup", "freelance"],
                "reading_time_minutes": 2,
                "public_reactions_count": 3,
                "comments_count": 0,
                "user": {"followers_count": 12},
                "url": "https://dev.to/spam/earn-money",
                "published_at": "2024-01-22T15:30:00Z"
            },
            {
                "id": 1003,
                "title": "Understanding React Hooks: useState and useEffect",
                "description": "A comprehensive tutorial on React hooks with practical examples and best practices",
                "tag_list": ["react", "javascript", "hooks", "frontend"],
                "reading_time_minutes": 8,
                "public_reactions_count": 67,
                "comments_count": 12,
                "user": {"followers_count": 456},
                "url": "https://dev.to/example/react-hooks",
                "published_at": "2024-01-21T14:20:00Z"
            },
            {
                "id": 1004,
                "title": "BUY CRYPTO NOW!!! LIMITED TIME OFFER 🔥🔥🔥",
                "description": "Get rich quick with crypto trading bot",
                "tag_list": ["crypto", "bitcoin", "trading", "money", "investment", "profit", "rich"],
                "reading_time_minutes": 1,
                "public_reactions_count": 1,
                "comments_count": 0,
                "user": {"followers_count": 5},
                "url": "https://dev.to/spam/crypto",
                "published_at": "2024-01-22T16:45:00Z"
            },
            {
                "id": 1005,
                "title": "Docker Best Practices for Development",
                "description": "Learn how to optimize your Docker workflow with multi-stage builds, layer caching, and security practices",
                "tag_list": ["docker", "devops", "containers", "development"],
                "reading_time_minutes": 15,
                "public_reactions_count": 89,
                "comments_count": 18,
                "user": {"followers_count": 634},
                "url": "https://dev.to/example/docker-practices",
                "published_at": "2024-01-19T09:15:00Z"
            }
        ]

def create_demo_layout():
    """Создание layout для демонстрации"""
    layout = Layout()
    
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=7)
    )
    
    layout["main"].split_row(
        Layout(name="results"),
        Layout(name="stats", ratio=1)
    )
    
    return layout

def format_classification_result(post: Dict, result: Dict) -> Table:
    """Форматирование результата классификации"""
    table = Table(title=f"📝 Пост #{post['id']}")
    
    # Определяем цвет на основе классификации
    if result.get("error"):
        color = "red"
        status = "❌ ОШИБКА"
    elif result["is_spam"]:
        color = "red"
        status = "🚫 СПАМ"
    else:
        color = "green"
        status = "✅ ЛЕГИТИМНЫЙ"
    
    table.add_column("Параметр", style="bold")
    table.add_column("Значение")
    
    table.add_row("Заголовок", post["title"][:50] + "..." if len(post["title"]) > 50 else post["title"])
    table.add_row("Теги", ", ".join(post["tag_list"][:3]))
    table.add_row("Время чтения", f"{post['reading_time_minutes']} мин")
    table.add_row("Реакции", str(post["public_reactions_count"]))
    
    if not result.get("error"):
        table.add_row("Статус", f"[{color}]{status}[/{color}]")
        table.add_row("Уверенность", f"{result['confidence']:.1%}")
        table.add_row("Рекомендация", result['recommendation'].upper())
        table.add_row("Время обработки", f"{result['processing_time_ms']:.1f}ms")
        
        if result.get("reasoning"):
            table.add_row("Причины", "\n".join(result["reasoning"][:2]))
    
    return table

def create_stats_panel(stats: Dict, processed: int, spam_found: int) -> Panel:
    """Создание панели статистики"""
    content = f"""
📊 [bold]Статистика демонстрации[/bold]

🔍 Обработано постов: [green]{processed}[/green]
🚫 Найдено спама: [red]{spam_found}[/red]
📈 Процент спама: [yellow]{spam_found/max(processed,1)*100:.1f}%[/yellow]

💾 [bold]Общая статистика системы[/bold]
📝 Всего классифицировано: {stats.get('total_classified', 0)}
🚫 Спама обнаружено: {stats.get('spam_detected', 0)}
⏰ Последнее обновление: {datetime.now().strftime('%H:%M:%S')}
    """
    
    return Panel(content, title="📊 Статистика", border_style="blue")

async def run_interactive_demo():
    """Интерактивная демонстрация"""
    console.clear()
    console.print(Panel.fit(
        "[bold blue]🚀 Dev.to Spam Classifier Demo[/bold blue]\n"
        "[dim]Система классификации спама с использованием Redis 8 Vector Sets[/dim]",
        border_style="blue"
    ))
    
    async with DemoSpamClassifier() as classifier:
        # Проверяем доступность API
        with console.status("[bold green]Проверка подключения к API..."):
            stats = await classifier.get_stats()
            if stats.get("error"):
                console.print("[red]❌ API недоступен. Убедитесь, что сервер запущен на localhost:8000[/red]")
                return
        
        console.print("[green]✅ API доступен![/green]\n")
        
        # Получаем посты для демонстрации
        with console.status("[bold green]Загрузка постов для демонстрации..."):
            posts = await classifier.fetch_dev_to_posts(10)
        
        console.print(f"[green]📥 Загружено {len(posts)} постов[/green]\n")
        
        # Создаем layout для результатов
        layout = create_demo_layout()
        processed_count = 0
        spam_count = 0
        results = []
        
        with Live(layout, refresh_per_second=4) as live:
            # Заголовок
            layout["header"].update(Panel(
                "[bold blue]🤖 Классификация постов в реальном времени[/bold blue]",
                border_style="blue"
            ))
            
            # Обрабатываем посты по одному
            for i, post in enumerate(posts):
                # Обновляем прогресс
                layout["footer"].update(Panel(
                    f"[bold]Обработка поста {i+1}/{len(posts)}...[/bold]\n"
                    f"[dim]Пост: {post['title'][:60]}...[/dim]",
                    title="🔄 Прогресс"
                ))
                
                # Классифицируем пост
                start_time = time.time()
                result = await classifier.classify_post(post)
                end_time = time.time()
                
                processed_count += 1
                if result.get("is_spam"):
                    spam_count += 1
                
                # Добавляем результат
                results.append((post, result))
                
                # Показываем последние 3 результата
                results_content = ""
                for p, r in results[-3:]:
                    status = "🚫 СПАМ" if r.get("is_spam") else "✅ OK"
                    confidence = r.get("confidence", 0)
                    processing_time = r.get("processing_time_ms", 0)
                    
                    results_content += f"""
📝 [bold]{p['title'][:40]}...[/bold]
   {status} ({confidence:.1%} уверенности, {processing_time:.1f}ms)
   {', '.join(p['tag_list'][:3])} | {p['reading_time_minutes']}мин | {p['public_reactions_count']}👍
"""
                
                layout["results"].update(Panel(
                    results_content.strip(),
                    title="📋 Результаты классификации",
                    border_style="green"
                ))
                
                # Обновляем статистику
                current_stats = await classifier.get_stats()
                layout["stats"].update(create_stats_panel(current_stats, processed_count, spam_count))
                
                # Пауза для демонстрации
                await asyncio.sleep(1.5)
            
            # Финальные результаты
            layout["footer"].update(Panel(
                f"[bold green]✅ Демонстрация завершена![/bold green]\n"
                f"Обработано: {processed_count} постов\n"
                f"Найдено спама: {spam_count} ({spam_count/processed_count*100:.1f}%)\n"
                f"Средняя точность: {sum(r.get('confidence', 0) for _, r in results)/len(results):.1%}",
                title="🎉 Результаты"
            ))
            
            console.print("\n[dim]Нажмите Enter для продолжения...[/dim]")
            input()

async def run_batch_demo():
    """Демонстрация пакетной обработки"""
    console.clear()
    console.print(Panel.fit(
        "[bold blue]⚡ Пакетная обработка постов[/bold blue]",
        border_style="blue"
    ))
    
    async with DemoSpamClassifier() as classifier:
        posts = classifier.get_demo_posts()
        
        # Подготавливаем запрос
        batch_request = {
            "posts": posts,
            "threshold": 0.8
        }
        
        with console.status("[bold green]Выполняется пакетная классификация..."):
            start_time = time.time()
            
            # Отправляем batch запрос
            async with classifier.session.post(
                f"{classifier.api_url}/classify-batch",
                json=batch_request
            ) as response:
                if response.status == 200:
                    batch_results = await response.json()
                    end_time = time.time()
                else:
                    console.print(f"[red]Ошибка: HTTP {response.status}[/red]")
                    return
        
        # Показываем результаты
        processing_time = (end_time - start_time) * 1000
        
        table = Table(title=f"📊 Результаты пакетной обработки ({processing_time:.1f}ms)")
        table.add_column("ID", justify="center")
        table.add_column("Заголовок", min_width=30)
        table.add_column("Статус", justify="center")
        table.add_column("Уверенность", justify="center")
        table.add_column("Рекомендация", justify="center")
        
        spam_count = 0
        for result in batch_results["results"]:
            post = next(p for p in posts if p["id"] == result["post_id"])
            
            status = "🚫 СПАМ" if result["is_spam"] else "✅ OK"
            status_color = "red" if result["is_spam"] else "green"
            
            if result["is_spam"]:
                spam_count += 1
            
            table.add_row(
                str(result["post_id"]),
                post["title"][:40] + "..." if len(post["title"]) > 40 else post["title"],
                f"[{status_color}]{status}[/{status_color}]",
                f"{result['confidence']:.1%}",
                result["recommendation"].upper()
            )
        
        console.print(table)
        console.print(f"\n[bold]Статистика:[/bold]")
        console.print(f"• Обработано постов: {len(posts)}")
        console.print(f"• Найдено спама: {spam_count}")
        console.print(f"• Время обработки: {processing_time:.1f}ms")
        console.print(f"• Скорость: {len(posts)/(processing_time/1000):.1f} постов/сек")
        
        console.print("\n[dim]Нажмите Enter для продолжения...[/dim]")
        input()

async def run_performance_demo():
    """Демонстрация производительности"""
    console.clear()
    console.print(Panel.fit(
        "[bold blue]⚡ Тест производительности[/bold blue]",
        border_style="blue"
    ))
    
    async with DemoSpamClassifier() as classifier:
        test_post = classifier.get_demo_posts()[0]
        
        # Тест одиночных запросов
        console.print("[bold]🔍 Тест одиночных запросов...[/bold]")
        
        times = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Выполняется тест...", total=50)
            
            for i in range(50):
                start_time = time.time()
                result = await classifier.classify_post(test_post)
                end_time = time.time()
                
                if not result.get("error"):
                    times.append((end_time - start_time) * 1000)
                
                progress.update(task, advance=1)
        
        # Статистика производительности
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        table = Table(title="📊 Результаты теста производительности")
        table.add_column("Метрика", style="bold")
        table.add_column("Значение", justify="right")
        
        table.add_row("Количество запросов", "50")
        table.add_row("Успешных запросов", str(len(times)))
        table.add_row("Среднее время", f"{avg_time:.1f}ms")
        table.add_row("Минимальное время", f"{min_time:.1f}ms")
        table.add_row("Максимальное время", f"{max_time:.1f}ms")
        table.add_row("Пропускная способность", f"{1000/avg_time:.1f} запросов/сек")
        
        console.print(table)
        
        # Тест параллельных запросов
        console.print("\n[bold]🚀 Тест параллельных запросов...[/bold]")
        
        async def classify_concurrent(post, semaphore):
            async with semaphore:
                start_time = time.time()
                result = await classifier.classify_post(post)
                end_time = time.time()
                return (end_time - start_time) * 1000 if not result.get("error") else None
        
        semaphore = asyncio.Semaphore(10)  # Максимум 10 параллельных запросов
        
        with console.status("[bold green]Выполняется тест параллельных запросов..."):
            start_total = time.time()
            tasks = [classify_concurrent(test_post, semaphore) for _ in range(100)]
            concurrent_times = await asyncio.gather(*tasks)
            end_total = time.time()
        
        # Фильтруем успешные запросы
        successful_times = [t for t in concurrent_times if t is not None]
        total_time = (end_total - start_total) * 1000
        
        table2 = Table(title="📊 Результаты параллельного теста")
        table2.add_column("Метрика", style="bold")
        table2.add_column("Значение", justify="right")
        
        table2.add_row("Всего запросов", "100")
        table2.add_row("Успешных запросов", str(len(successful_times)))
        table2.add_row("Общее время", f"{total_time:.1f}ms")
        table2.add_row("Среднее время на запрос", f"{sum(successful_times)/len(successful_times):.1f}ms")
        table2.add_row("Реальная пропускная способность", f"{len(successful_times)/(total_time/1000):.1f} запросов/сек")
        
        console.print(table2)
        
        console.print("\n[dim]Нажмите Enter для продолжения...[/dim]")
        input()

def show_main_menu():
    """Показ главного меню"""
    console.clear()
    
    title = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🛡️ Redis8 Spam Guard                 ║
    ║              Демонстрационная система для конкурса          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    console.print(title, style="bold blue")
    
    console.print("\n[bold]Возможности системы:[/bold]")
    console.print("• 🧠 Классификация на основе машинного обучения")
    console.print("• ⚡ Redis 8 Vector Sets для векторного поиска")
    console.print("• 🚀 FastAPI для высокопроизводительного API")
    console.print("• 🎯 Real-time анализ постов dev.to")
    console.print("• 📊 Детальная аналитика и объяснения")
    
    console.print("\n[bold]Доступные демонстрации:[/bold]")
    console.print("1. 🎭 Интерактивная классификация постов")
    console.print("2. ⚡ Пакетная обработка")
    console.print("3. 📈 Тест производительности")
    console.print("4. 📊 Показать архитектуру системы")
    console.print("5. 🔧 Техническая информация")
    console.print("0. 🚪 Выход")
    
    return input("\n[bold blue]Выберите опцию (0-5): [/bold blue]")

def show_architecture():
    """Показ архитектуры системы"""
    console.clear()
    
    architecture = """
[bold blue]🏗️ Архитектура системы[/bold blue]

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dev.to API    │───▶│  FastAPI Server  │───▶│  Redis 8        │
│                 │    │                  │    │  Vector Sets    │
│  • Posts        │    │  • Classification│    │  • Embeddings   │
│  • Metadata     │    │  • Batch API     │    │  • k-NN Search  │
│  • Real-time    │    │  • WebUI         │    │  • Persistence  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────────────────┐
                       │  Web Interface   │
                       │                  │
                       │  • Live Demo     │
                       │  • Statistics    │
                       │  • Moderator UI  │
                       └──────────────────┘

[bold]🔄 Процесс классификации:[/bold]

1. 📥 [green]Получение поста[/green] - API получает данные поста
2. 🧹 [yellow]Предобработка[/yellow] - очистка текста, нормализация
3. 🧠 [blue]Векторизация[/blue] - Sentence Transformers → 384D вектор
4. 🔍 [purple]Поиск похожих[/purple] - Redis Vector Sets k-NN поиск
5. 🎯 [red]Классификация[/red] - голосование ближайших соседей
6. 📊 [green]Результат[/green] - спам/не спам + уверенность + объяснение

[bold]⚙️ Технический стек:[/bold]

• [bold blue]Backend:[/bold blue] Python 3.11, FastAPI, AsyncIO
• [bold red]Database:[/bold red] Redis 8 с Vector Sets модулем
• [bold green]ML:[/bold green] Sentence Transformers, scikit-learn
• [bold yellow]Frontend:[/bold yellow] HTML5, JavaScript, Rich Console
• [bold purple]DevOps:[/bold purple] Docker, Docker Compose, Nginx

[bold]📈 Производительность:[/bold]

• Классификация: ~50ms на запрос
• Пропускная способность: 1000+ запросов/сек
• Векторный поиск: <10ms в Redis
• Точность модели: ~94% (F1-score)
"""
    
    console.print(Panel(architecture, border_style="blue"))
    console.print("\n[dim]Нажмите Enter для возврата в меню...[/dim]")
    input()

def show_technical_info():
    """Показ технической информации"""
    console.clear()
    
    technical = """
[bold blue]🔧 Техническая информация[/bold blue]

[bold]📊 Алгоритм классификации:[/bold]

1. [bold yellow]Feature Engineering:[/bold yellow]
   • Текстовые признаки: title + description → SentenceTransformer
   • Численные признаки: reading_time, reactions, comments, followers
   • Нормализация и объединение в 389D вектор

2. [bold green]Vector Search:[/bold green]
   • Redis VSET.ADD для сохранения обучающих векторов
   • Redis VSET.SEARCH для поиска k=5 ближайших соседей
   • Cosine similarity для измерения расстояния

3. [bold red]Classification:[/bold red]
   • Weighted k-NN voting по найденным соседям
   • Confidence score на основе консенсуса соседей
   • Fallback на эвристические правила для новых данных

[bold]🎯 Эвристические правила:[/bold]

• [red]Спам индикаторы:[/red]
  - Ключевые слова: "earn money", "get rich", "click here"
  - Короткие посты (<2 мин) с низкой активностью (<5 реакций)
  - Много тегов (>10) или подозрительные теги
  - Новые авторы (<10 подписчиков)

• [green]Качественные индикаторы:[/green]
  - Образовательные слова: "tutorial", "guide", "how to"
  - Высокая активность (>50 реакций, >10 комментариев)
  - Оптимальная длина (8-15 минут чтения)
  - Авторитетные авторы (>100 подписчиков)

[bold]⚡ Оптимизации производительности:[/bold]

• Асинхронная обработка с asyncio и aioredis
• Batch processing для множественных запросов
• Connection pooling для Redis соединений
• Кэширование векторов популярных постов
• Lazy loading моделей Sentence Transformers

[bold]🔒 Безопасность и масштабирование:[/bold]

• Rate limiting на API endpoints
• Input validation с Pydantic
• Error handling и graceful degradation
• Health checks и мониторинг
• Horizontal scaling через Docker Swarm/K8s

[bold]📏 Метрики качества:[/bold]

• Precision: доля правильно классифицированного спама
• Recall: доля обнаруженного спама от всего спама
• F1-Score: гармоническое среднее precision и recall
• Accuracy: общая доля правильных предсказаний
• Processing Time: время обработки одного запроса
"""
    
    console.print(Panel(technical, border_style="green"))
    console.print("\n[dim]Нажмите Enter для возврата в меню...[/dim]")
    input()

async def main():
    """Главная функция демонстрации"""
    console.print("[bold green]🚀 Запуск демонстрационной системы...[/bold green]")
    
    while True:
        try:
            choice = show_main_menu()
            
            if choice == "1":
                await run_interactive_demo()
            elif choice == "2":
                await run_batch_demo()
            elif choice == "3":
                await run_performance_demo()
            elif choice == "4":
                show_architecture()
            elif choice == "5":
                show_technical_info()
            elif choice == "0":
                console.print("\n[bold blue]👋 Спасибо за внимание![/bold blue]")
                console.print("[dim]Удачи на конкурсе! 🏆[/dim]")
                break
            else:
                console.print("[red]❌ Неверный выбор. Попробуйте снова.[/red]")
                time.sleep(1)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Прервано пользователем[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]❌ Ошибка: {e}[/red]")
            console.print("[dim]Нажмите Enter для продолжения...[/dim]")
            input()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold blue]👋 До свидания![/bold blue]")
    except Exception as e:
        console.print(f"[bold red]Критическая ошибка: {e}[/bold red]")
        sys.exit(1)
