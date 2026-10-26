"""Benchmark schemas for schemarepair.

Eight Pydantic v2 schemas spanning the shapes that dominate real
structured-output workloads: flat records, enums, nested objects, lists of
objects, numeric fields with units in the source text, and geo coordinates.

All schemas forbid extra fields: a hallucinated key is a validation failure,
which is the behaviour production APIs with strict contracts want.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PersonRecord(_Base):
    name: str
    age: int = Field(ge=0, le=120)
    email: str
    is_member: bool


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    PKR = "PKR"


class ProductItem(_Base):
    title: str
    price: float = Field(ge=0)
    currency: Currency
    in_stock: bool
    tags: list[str]


class Priority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"


class Appointment(_Base):
    patient: str
    doctor: str
    date: str  # ISO date, kept as string: format discipline is part of the test
    start_time: str  # "HH:MM" 24h
    reason: str
    priority: Priority


class InvoiceItem(_Base):
    item: str
    qty: int = Field(ge=1)
    unit_price: float = Field(ge=0)


class Invoice(_Base):
    invoice_id: str
    customer: str
    items: list[InvoiceItem]
    total: float = Field(ge=0)


class SensorStatus(str, Enum):
    ok = "ok"
    warn = "warn"
    error = "error"


class SensorReading(_Base):
    device_id: str
    temperature_c: float
    humidity_pct: int = Field(ge=0, le=100)
    battery_pct: int = Field(ge=0, le=100)
    status: SensorStatus


class BookRecord(_Base):
    title: str
    author: str
    year: int = Field(ge=1450, le=2100)
    genres: list[str]
    rating: float = Field(ge=0, le=5)


class TicketCategory(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    other = "other"


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class SupportTicket(_Base):
    ticket_id: str
    subject: str
    category: TicketCategory
    sentiment: Sentiment
    resolved: bool
    tags: list[str]


class GeoPlace(_Base):
    name: str
    country: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    population: int = Field(ge=0)


SCHEMAS: dict[str, type[BaseModel]] = {
    "person_record": PersonRecord,
    "product_item": ProductItem,
    "appointment": Appointment,
    "invoice": Invoice,
    "sensor_reading": SensorReading,
    "book_record": BookRecord,
    "support_ticket": SupportTicket,
    "geo_place": GeoPlace,
}
