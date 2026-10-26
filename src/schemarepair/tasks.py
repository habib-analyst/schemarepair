"""Benchmark tasks: ground-truth records rendered as messy source text.

Each task gives the model a schema plus a record written the way records
actually arrive in pipelines: booleans as yes/no, money with currency symbols,
numbers with thousands separators, coordinates with hemisphere letters,
ratings as "4.5 out of 5". The ground-truth dict is the exact object the
schema should validate to, which lets the benchmark measure not only
*conformance* (does it validate?) but *field accuracy* (are the values the
ones from the record?) — a repaired output with wrong values is a silent
failure, and this benchmark refuses to count it as a clean win.

Records are generated programmatically from fixed pools with a fixed seed,
so the corpus is identical on every run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schemas import SCHEMAS

SEED = 20261026
PER_SCHEMA = 24

FIRST = ["Sara", "Ahmed", "Priya", "Jonas", "Mei", "Carlos", "Aisha", "Tom",
         "Nadia", "Omar", "Ingrid", "Kenji", "Lucia", "Pavel", "Zainab", "Ruth"]
LAST = ["Iqbal", "Khan", "Sharma", "Weber", "Chen", "Garcia", "Bello", "Novak",
        "Haddad", "Aziz", "Larsen", "Tanaka", "Rossi", "Petrov", "Malik", "Berg"]
CITIES = [("Lahore", "Pakistan", 31.5497, 74.3436, 13095000),
          ("Berlin", "Germany", 52.5200, 13.4050, 3645000),
          ("Toronto", "Canada", 43.6532, -79.3832, 2930000),
          ("Sydney", "Australia", -33.8688, 151.2093, 5312000),
          ("Nairobi", "Kenya", -1.2921, 36.8219, 4844000),
          ("Osaka", "Japan", 34.6937, 135.5023, 2753000),
          ("Lima", "Peru", -12.0464, -77.0428, 10195000),
          ("Oslo", "Norway", 59.9139, 10.7522, 709000)]
PRODUCTS = ["wireless earbuds", "standing desk", "trail running shoes",
            "ceramic pour-over set", "mechanical keyboard", "insulated bottle",
            "linen bedding set", "usb-c hub", "desk lamp", "yoga mat"]
BOOKS = [("The Design of Everyday Things", "Don Norman", 1988),
         ("Thinking, Fast and Slow", "Daniel Kahneman", 2011),
         ("A Brief History of Time", "Stephen Hawking", 1988),
         ("The Pragmatic Programmer", "David Thomas", 1999),
         ("Sapiens", "Yuval Noah Harari", 2011),
         ("Deep Learning", "Ian Goodfellow", 2016),
         ("The Midnight Library", "Matt Haig", 2020),
         ("Clean Code", "Robert C. Martin", 2008)]
GENRES = ["fiction", "science", "history", "technology", "biography",
          "philosophy", "design", "mathematics"]
DOCTORS = ["Dr. Farah Aziz", "Dr. Henry Cole", "Dr. Yuki Mori", "Dr. Elena Petrova"]
REASONS = ["annual checkup", "follow-up on blood work", "knee pain review",
           "vaccination", "migraine consultation", "skin screening"]
SUBJECTS = ["charged twice for march invoice", "app crashes on export",
            "cannot reset password", "missing refund for order 881",
            "sync fails on android", "invoice address is wrong",
            "login loop after update", "export missing february rows"]
ITEMS = [("notebook", 3.50), ("pen set", 6.25), ("backpack", 42.00),
         ("water bottle", 12.75), ("desk organizer", 18.40), ("cable kit", 9.99),
         ("mouse pad", 7.50), ("planter", 14.20)]


@dataclass
class Task:
    task_id: str
    schema_name: str
    record_text: str
    ground_truth: dict

    @property
    def schema_cls(self):
        return SCHEMAS[self.schema_name]

    def messages(self, variant: str = "strict") -> list[dict[str, str]]:
        """Chat messages for this task.

        Two prompting regimes, both ubiquitous in real apps: ``strict``
        (an explicit JSON-only system prompt with the schema) and ``plain``
        (a generic assistant system prompt, schema only in the user turn).
        The benchmark alternates them and reports validity per regime.
        """
        import json

        schema_json = json.dumps(self.schema_cls.model_json_schema(), indent=1)
        if variant == "plain":
            system = "You are a helpful assistant."
            user = (
                f"Convert this record to JSON matching this schema:\n"
                f"{schema_json}\n\nRecord:\n{self.record_text}"
            )
        else:
            system = (
                "You are a data extraction engine. You convert records into "
                "JSON that matches a given schema. Respond with a single "
                "JSON object only - no markdown, no explanation."
            )
            user = (
                f"Schema (JSON Schema):\n{schema_json}\n\n"
                f"Record:\n{self.record_text}\n\n"
                "Return one JSON object matching the schema, using exactly "
                "the values from the record."
            )
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]


def _person(rng: random.Random) -> tuple[str, dict]:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    age = rng.randint(19, 84)
    email = f"{name.lower().replace(' ', '.')}@example.com"
    member = rng.random() < 0.5
    text = (f"Full name: {name}; age: {age} years; email address: {email}; "
            f"club member: {'yes' if member else 'no'}")
    gt = {"name": name, "age": age, "email": email, "is_member": member}
    return text, gt


def _product(rng: random.Random) -> tuple[str, dict]:
    title = rng.choice(PRODUCTS)
    price = round(rng.uniform(5, 400), 2)
    cur = rng.choice(["USD", "EUR", "GBP", "PKR"])
    sym = {"USD": "$", "EUR": "EUR ", "GBP": "GBP ", "PKR": "Rs "}[cur]
    stock = rng.random() < 0.6
    tags = rng.sample(["sale", "new", "eco", "pro", "limited", "refurb"],
                      k=rng.randint(1, 3))
    text = (f"Product: {title}; price: {sym}{price:.2f}; currency code: {cur}; "
            f"in stock: {'yes' if stock else 'no'}; tags: {', '.join(tags)}")
    gt = {"title": title, "price": price, "currency": cur,
          "in_stock": stock, "tags": tags}
    return text, gt


def _appointment(rng: random.Random) -> tuple[str, dict]:
    patient = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    doctor = rng.choice(DOCTORS)
    date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    time = f"{rng.randint(8, 17):02d}:{rng.choice(['00', '15', '30', '45'])}"
    reason = rng.choice(REASONS)
    prio = rng.choice(["low", "normal", "high", "urgent"])
    text = (f"Patient: {patient}; doctor: {doctor}; date: {date}; "
            f"start time: {time}; reason: {reason}; priority: {prio}")
    gt = {"patient": patient, "doctor": doctor, "date": date,
          "start_time": time, "reason": reason, "priority": prio}
    return text, gt


def _invoice(rng: random.Random) -> tuple[str, dict]:
    iid = f"INV-{rng.randint(1000, 9999)}"
    customer = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    picks = rng.sample(ITEMS, k=rng.randint(2, 3))
    items, lines, total = [], [], 0.0
    for name, unit in picks:
        qty = rng.randint(1, 6)
        items.append({"item": name, "qty": qty, "unit_price": unit})
        lines.append(f"{qty} x {name} at ${unit:.2f}")
        total += qty * unit
    total = round(total, 2)
    text = (f"Invoice {iid} for customer {customer}. Lines: "
            + "; ".join(lines) + f". Total due: ${total:.2f}")
    gt = {"invoice_id": iid, "customer": customer, "items": items,
          "total": total}
    return text, gt


def _sensor(rng: random.Random) -> tuple[str, dict]:
    dev = f"sensor-{rng.randint(100, 999)}"
    temp = round(rng.uniform(-5, 42), 1)
    hum = rng.randint(15, 95)
    batt = rng.randint(5, 100)
    status = rng.choice(["ok", "warn", "error"])
    text = (f"Device {dev} reports temperature {temp} degrees C, humidity "
            f"{hum} percent, battery {batt} percent, status {status}")
    gt = {"device_id": dev, "temperature_c": temp, "humidity_pct": hum,
          "battery_pct": batt, "status": status}
    return text, gt


def _book(rng: random.Random) -> tuple[str, dict]:
    title, author, year = rng.choice(BOOKS)
    genres = rng.sample(GENRES, k=rng.randint(1, 3))
    rating = round(rng.uniform(2.5, 5.0), 1)
    text = (f"Book: \"{title}\" by {author}, published {year}; genres: "
            f"{', '.join(genres)}; average rating {rating} out of 5")
    gt = {"title": title, "author": author, "year": year,
          "genres": genres, "rating": rating}
    return text, gt


def _ticket(rng: random.Random) -> tuple[str, dict]:
    tid = f"T-{rng.randint(10000, 99999)}"
    subject = rng.choice(SUBJECTS)
    cat = rng.choice(["billing", "technical", "account", "other"])
    sent = rng.choice(["positive", "neutral", "negative"])
    resolved = rng.random() < 0.5
    tags = rng.sample(["urgent", "vip", "follow-up", "bug", "billing",
                       "mobile", "web"], k=rng.randint(1, 3))
    text = (f"Ticket {tid}: subject \"{subject}\"; category {cat}; customer "
            f"sentiment {sent}; resolved: {'yes' if resolved else 'no'}; "
            f"tags: {', '.join(tags)}")
    gt = {"ticket_id": tid, "subject": subject, "category": cat,
          "sentiment": sent, "resolved": resolved, "tags": tags}
    return text, gt


def _geo(rng: random.Random) -> tuple[str, dict]:
    name, country, lat, lon, pop = rng.choice(CITIES)
    lat_s = f"{abs(lat):.4f} degrees {'N' if lat >= 0 else 'S'}"
    lon_s = f"{abs(lon):.4f} degrees {'E' if lon >= 0 else 'W'}"
    text = (f"Place: {name}, {country}; latitude {lat_s}; longitude {lon_s}; "
            f"population {pop:,}")
    gt = {"name": name, "country": country, "latitude": lat,
          "longitude": lon, "population": pop}
    return text, gt


_BUILDERS = {
    "person_record": _person,
    "product_item": _product,
    "appointment": _appointment,
    "invoice": _invoice,
    "sensor_reading": _sensor,
    "book_record": _book,
    "support_ticket": _ticket,
    "geo_place": _geo,
}


def build_tasks(per_schema: int = PER_SCHEMA) -> list[Task]:
    rng = random.Random(SEED)
    tasks: list[Task] = []
    for schema_name, builder in _BUILDERS.items():
        for i in range(per_schema):
            text, gt = builder(rng)
            tasks.append(Task(task_id=f"{schema_name}-{i:02d}",
                              schema_name=schema_name,
                              record_text=text, ground_truth=gt))
    return tasks
