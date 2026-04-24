"""
CrewAI Multi-Agent Demo: Travel Planning System
================================================

This demonstrates CrewAI's task-based multi-agent orchestration by planning
a 5-day trip to Iceland. Each agent has specialized tools that return
structured travel data, which the LLM then uses to create recommendations.

Agents:
1. FlightAgent - Flight Specialist (researches flight options)
2. HotelAgent - Accommodation Specialist (finds hotels)
3. ItineraryAgent - Travel Planner (creates day-by-day itineraries)
4. LocalExpert - Local Culture & Safety Specialist (adds local guidance)
5. BudgetAgent - Financial Advisor (analyzes total costs)

Configuration:
- Uses shared configuration from the root .env file
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from crewai import Agent, Task, Crew

# Add parent directory to path to import shared_config
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared configuration
from shared_config import Config, validate_config


# ============================================================================
# RESEARCH PACKETS
# ============================================================================

def search_flight_prices(destination: str, departure_city: str = "New York") -> str:
    """
    Search for flight prices and options to a destination.
    Returns current flight information from major booking sites.
    """
    # Static flight data simulating real search results
    flights_data = {
        "Iceland": [
            {"airline": "Icelandair", "route": f"{departure_city} (JFK) → Reykjavik (KEF)", "type": "Direct", "duration": "5h 30m", "price": "$485 round-trip", "schedule": "Daily departures at 8:30 PM"},
            {"airline": "Delta Air Lines", "route": f"{departure_city} (JFK) → Reykjavik (KEF)", "type": "Direct", "duration": "5h 45m", "price": "$612 round-trip", "schedule": "Mon/Wed/Fri/Sat at 10:15 PM"},
            {"airline": "PLAY Airlines", "route": f"{departure_city} (SWF) → Reykjavik (KEF)", "type": "Direct (budget)", "duration": "5h 20m", "price": "$349 round-trip", "schedule": "Tue/Thu/Sun at 11:00 PM, no checked bags included"},
            {"airline": "Norse Atlantic", "route": f"{departure_city} (JFK) → Reykjavik (KEF)", "type": "Direct (budget)", "duration": "5h 35m", "price": "$389 round-trip", "schedule": "Daily at 7:45 PM, carry-on only"},
            {"airline": "British Airways", "route": f"{departure_city} (JFK) → London (LHR) → Reykjavik (KEF)", "type": "1 stop", "duration": "11h 20m", "price": "$578 round-trip", "schedule": "Daily, 4h layover in London"},
        ],
        "default": [
            {"airline": "United Airlines", "route": f"{departure_city} → {destination}", "type": "Direct", "duration": "8h 00m", "price": "$650 round-trip", "schedule": "Daily"},
            {"airline": "Delta Air Lines", "route": f"{departure_city} → {destination}", "type": "1 stop", "duration": "11h 30m", "price": "$520 round-trip", "schedule": "Daily"},
            {"airline": "Budget Air", "route": f"{departure_city} → {destination}", "type": "Direct (budget)", "duration": "8h 15m", "price": "$410 round-trip", "schedule": "Mon/Wed/Fri"},
        ],
    }

    key = "Iceland" if "iceland" in destination.lower() or "reykjavik" in destination.lower() else "default"
    results = flights_data[key]

    output = f"Flight Search Results: {departure_city} → {destination}\n"
    output += f"{'='*60}\n"
    for i, flight in enumerate(results, 1):
        output += f"\n{i}. {flight['airline']}\n"
        output += f"   Route: {flight['route']}\n"
        output += f"   Type: {flight['type']} | Duration: {flight['duration']}\n"
        output += f"   Price: {flight['price']}\n"
        output += f"   Schedule: {flight['schedule']}\n"
    output += f"\nNote: Prices as of January 2026. Book 6-8 weeks in advance for best rates."
    return output


def search_hotel_options(location: str, check_in_date: str) -> str:
    """
    Search for hotel options in a location.
    Returns current hotel availability and pricing information.
    """
    # Static hotel data simulating real search results
    hotels_data = {
        "Reykjavik": [
            {"name": "CenterHotel Midgardur", "stars": 4, "rating": 8.7, "reviews": 2341, "price": "$189/night", "location": "Downtown Reykjavik, 2 min walk to Hallgrimskirkja", "amenities": "Free WiFi, breakfast included, restaurant, bar, 24h front desk", "style": "Mid-range"},
            {"name": "Canopy by Hilton Reykjavik", "stars": 4, "rating": 9.1, "reviews": 1876, "price": "$265/night", "location": "Smidjustigur 4, city center", "amenities": "Rooftop bar, gym, spa, restaurant, free WiFi, heated floors", "style": "Upscale"},
            {"name": "Kex Hostel", "stars": 2, "rating": 8.2, "reviews": 3102, "price": "$85/night (private room)", "location": "Skulagata 28, harbor district", "amenities": "Shared lounge, bar, free WiFi, bike rental, live music events", "style": "Budget"},
            {"name": "Hotel Borg by Keahotels", "stars": 5, "rating": 9.3, "reviews": 1204, "price": "$385/night", "location": "Posthusstraeti 11, overlooking Austurvollur Square", "amenities": "Art deco design, spa, fine dining, butler service, airport transfer", "style": "Luxury"},
            {"name": "Reykjavik Lights Hotel", "stars": 3, "rating": 8.4, "reviews": 1567, "price": "$145/night", "location": "Sudurlandsbraut 12, 10 min bus to center", "amenities": "Free parking, breakfast buffet, northern lights wake-up call, free WiFi", "style": "Mid-range"},
        ],
        "default": [
            {"name": "City Center Hotel", "stars": 4, "rating": 8.5, "reviews": 1200, "price": "$175/night", "location": f"Downtown {location}", "amenities": "Free WiFi, breakfast, gym", "style": "Mid-range"},
            {"name": "Budget Inn", "stars": 2, "rating": 7.8, "reviews": 890, "price": "$79/night", "location": f"Near transit, {location}", "amenities": "Free WiFi, shared kitchen", "style": "Budget"},
            {"name": "Grand Luxury Resort", "stars": 5, "rating": 9.2, "reviews": 650, "price": "$350/night", "location": f"Premium district, {location}", "amenities": "Spa, pool, fine dining, concierge", "style": "Luxury"},
        ],
    }

    key = "Reykjavik" if "reykjavik" in location.lower() or "iceland" in location.lower() else "default"
    results = hotels_data[key]

    output = f"Hotel Search Results: {location} (check-in: {check_in_date})\n"
    output += f"{'='*60}\n"
    for i, hotel in enumerate(results, 1):
        output += f"\n{i}. {hotel['name']} {'⭐' * hotel['stars']}\n"
        output += f"   Rating: {hotel['rating']}/10 ({hotel['reviews']} reviews) | Style: {hotel['style']}\n"
        output += f"   Price: {hotel['price']}\n"
        output += f"   Location: {hotel['location']}\n"
        output += f"   Amenities: {hotel['amenities']}\n"
    output += f"\nNote: Prices for January 2026. 5-night stay recommended for best value."
    return output


def search_attractions_activities(destination: str) -> str:
    """
    Search for attractions and activities in a destination.
    Returns popular sites, tours, and experiences with pricing.
    """
    # Static attractions data simulating real search results
    attractions_data = {
        "Iceland": [
            {"name": "Golden Circle Tour", "type": "Day Tour", "duration": "8 hours", "price": "$85/person", "description": "Visit Thingvellir National Park, Geysir geothermal area, and Gullfoss waterfall. Includes hotel pickup.", "rating": 4.8},
            {"name": "Blue Lagoon", "type": "Spa/Attraction", "duration": "2-3 hours", "price": "$75-115/person (Comfort-Premium)", "description": "World-famous geothermal spa with silica mud masks and in-water bar. Book 2+ weeks in advance.", "rating": 4.5},
            {"name": "South Coast & Black Sand Beach", "type": "Day Tour", "duration": "10 hours", "price": "$95/person", "description": "Seljalandsfoss and Skogafoss waterfalls, Reynisfjara black sand beach, Vik village.", "rating": 4.9},
            {"name": "Northern Lights Tour", "type": "Evening Tour", "duration": "3-4 hours", "price": "$65/person", "description": "Guided bus tour to dark-sky locations. Free rebooking if no lights visible. Best Oct-Mar.", "rating": 4.3},
            {"name": "Glacier Hiking on Solheimajokull", "type": "Adventure", "duration": "3 hours", "price": "$110/person", "description": "Guided glacier walk with crampons and ice axes provided. No experience needed. Min age 8.", "rating": 4.9},
            {"name": "Whale Watching from Reykjavik", "type": "Boat Tour", "duration": "3 hours", "price": "$79/person", "description": "See humpback whales, dolphins, and puffins (summer). Warm overalls provided. 95% sighting rate.", "rating": 4.6},
            {"name": "Snorkeling in Silfra Fissure", "type": "Adventure", "duration": "3 hours", "price": "$145/person", "description": "Snorkel between tectonic plates in crystal-clear glacial water (2°C). Dry suit provided.", "rating": 4.8},
            {"name": "Hallgrimskirkja Church Tower", "type": "Landmark", "duration": "30 min", "price": "$12/person", "description": "Iconic Reykjavik church with elevator to observation deck. Panoramic city views.", "rating": 4.4},
            {"name": "Reykjavik Food Walk", "type": "Food Tour", "duration": "3 hours", "price": "$95/person", "description": "6 tastings including fermented shark, lamb soup, skyr, and craft beer. Small groups.", "rating": 4.7},
            {"name": "Ice Cave Exploration (Vatnajokull)", "type": "Adventure", "duration": "Full day (12h from Reykjavik)", "price": "$250/person", "description": "Visit naturally-formed blue ice caves inside Europe's largest glacier. Nov-Mar only.", "rating": 4.9},
        ],
        "default": [
            {"name": "City Walking Tour", "type": "Tour", "duration": "3 hours", "price": "$40/person", "description": f"Guided walking tour of {destination} highlights.", "rating": 4.5},
            {"name": "Local Food Experience", "type": "Food Tour", "duration": "2.5 hours", "price": "$75/person", "description": "Taste local cuisine with a knowledgeable guide.", "rating": 4.7},
            {"name": "Nature Day Trip", "type": "Day Tour", "duration": "8 hours", "price": "$95/person", "description": f"Full-day excursion to natural attractions near {destination}.", "rating": 4.6},
        ],
    }

    key = "Iceland" if "iceland" in destination.lower() or "reykjavik" in destination.lower() else "default"
    results = attractions_data[key]

    output = f"Attractions & Activities: {destination}\n"
    output += f"{'='*60}\n"
    for i, item in enumerate(results, 1):
        output += f"\n{i}. {item['name']} ({item['type']})\n"
        output += f"   Duration: {item['duration']} | Price: {item['price']} | Rating: {item['rating']}/5.0\n"
        output += f"   {item['description']}\n"
    output += f"\nTip: Book popular tours 1-2 weeks in advance, especially Golden Circle and Blue Lagoon."
    return output


def search_travel_costs(destination: str) -> str:
    """
    Search for travel costs and budgeting information.
    Returns current pricing for meals, activities, and transportation.
    """
    # Static cost data simulating real search results
    costs_data = {
        "Iceland": {
            "currency": "Icelandic Krona (ISK). 1 USD = ~137 ISK. Credit cards accepted everywhere.",
            "meals": [
                {"category": "Budget", "examples": "Hot dogs (Baejarins Beztu), gas station sandwiches, grocery store meals", "avg_cost": "$15-25/meal"},
                {"category": "Mid-range", "examples": "Cafe meals, fish & chips, lamb soup at local restaurants", "avg_cost": "$30-50/meal"},
                {"category": "Fine dining", "examples": "Grillid, Dill (Michelin), Matur og Drykkur", "avg_cost": "$80-150/meal"},
            ],
            "transport": [
                {"type": "Airport bus (Flybus)", "cost": "$28 one-way to Reykjavik"},
                {"type": "Reykjavik city bus (Straeto)", "cost": "$4.20/ride or $24/3-day pass"},
                {"type": "Rental car (compact)", "cost": "$65-95/day (add $15/day for insurance)"},
                {"type": "Rental car (4WD, for highlands)", "cost": "$120-180/day"},
                {"type": "Taxi (Reykjavik)", "cost": "$20-35 within city center"},
                {"type": "Domestic flight to Akureyri", "cost": "$120-180 round-trip"},
            ],
            "daily_budgets": [
                {"level": "Budget", "per_day": "$150-200/day", "notes": "Hostel, bus tours, grocery meals, free attractions"},
                {"level": "Mid-range", "per_day": "$300-400/day", "notes": "3-4 star hotel, guided tours, restaurant meals"},
                {"level": "Luxury", "per_day": "$600+/day", "notes": "5-star hotel, private tours, fine dining, spa visits"},
            ],
            "tips": [
                "Tap water is free and excellent — no need to buy bottled water",
                "Happy hour (15:00-18:00) cuts drink prices by 40-50%",
                "Bonus/Kronan supermarkets are cheapest for groceries",
                "Free attractions: Hallgrimskirkja exterior, Harpa concert hall, city walking paths",
                "Gas is expensive (~$8.50/gallon) — factor into rental car budget",
            ],
        },
        "default": {
            "currency": "Local currency. Credit cards widely accepted.",
            "meals": [
                {"category": "Budget", "examples": "Street food, fast casual", "avg_cost": "$10-20/meal"},
                {"category": "Mid-range", "examples": "Sit-down restaurants", "avg_cost": "$25-45/meal"},
                {"category": "Fine dining", "examples": "Upscale restaurants", "avg_cost": "$60-120/meal"},
            ],
            "transport": [
                {"type": "Public transit", "cost": "$3-5/ride"},
                {"type": "Taxi", "cost": "$15-30 within city"},
                {"type": "Rental car", "cost": "$50-80/day"},
            ],
            "daily_budgets": [
                {"level": "Budget", "per_day": "$100-150/day", "notes": "Hostel, public transit, street food"},
                {"level": "Mid-range", "per_day": "$200-300/day", "notes": "Hotel, tours, restaurants"},
                {"level": "Luxury", "per_day": "$500+/day", "notes": "Luxury hotel, private tours, fine dining"},
            ],
            "tips": ["Research local discount passes", "Eat where locals eat", "Book tours in advance for better rates"],
        },
    }

    key = "Iceland" if "iceland" in destination.lower() or "reykjavik" in destination.lower() else "default"
    data = costs_data[key]

    output = f"Travel Cost Guide: {destination}\n"
    output += f"{'='*60}\n"
    output += f"\nCurrency: {data['currency']}\n"

    output += f"\n--- Meal Costs ---\n"
    for meal in data["meals"]:
        output += f"  {meal['category']}: {meal['avg_cost']} ({meal['examples']})\n"

    output += f"\n--- Transportation ---\n"
    for t in data["transport"]:
        output += f"  {t['type']}: {t['cost']}\n"

    output += f"\n--- Daily Budget Estimates (per person) ---\n"
    for b in data["daily_budgets"]:
        output += f"  {b['level']}: {b['per_day']} — {b['notes']}\n"

    output += f"\n--- Money-Saving Tips ---\n"
    for i, tip in enumerate(data["tips"], 1):
        output += f"  {i}. {tip}\n"

    return output


def search_local_tips(destination: str) -> str:
    """
    Search for local customs, safety tips, and practical travel advice.
    Returns destination-specific guidance that can improve the itinerary and budget.
    """
    local_tips_data = {
        "Iceland": {
            "customs": [
                "Swimming etiquette matters: shower thoroughly without a swimsuit before entering public pools or lagoons.",
                "Tipping is not expected because service charges and fair wages are already built into prices.",
                "Remove muddy shoes when entering guesthouses or homes, especially during wet winter months.",
            ],
            "safety": [
                "Check SafeTravel.is and local weather alerts before long drives because wind and road closures change quickly.",
                "Winter daylight is limited, so plan major outdoor activities earlier in the day.",
                "Stay on marked paths near geothermal areas and black sand beaches due to sudden hazards and strong waves.",
            ],
            "money": [
                "Credit cards are accepted almost everywhere, even for very small purchases.",
                "Groceries from Bonus or Kronan are far cheaper than restaurant meals for breakfast or snacks.",
                "Book Blue Lagoon and popular tours early to avoid higher last-minute prices.",
            ],
            "packing": [
                "Pack waterproof outer layers, insulated boots, gloves, and a windproof hat.",
                "Bring a reusable water bottle because Icelandic tap water is excellent and free.",
            ],
        },
        "default": {
            "customs": [
                "Learn a few local etiquette basics before arrival to avoid common tourist mistakes.",
                "Confirm whether tipping is customary so you can budget accurately.",
            ],
            "safety": [
                "Review local transport, weather, and neighborhood safety guidance before finalizing daily plans.",
                "Keep digital and physical copies of passports, booking confirmations, and emergency contacts.",
            ],
            "money": [
                "Use local grocery stores and transit passes when available to reduce daily expenses.",
                "Book high-demand attractions in advance to avoid premium same-day pricing.",
            ],
            "packing": [
                "Pack layers and comfortable walking shoes suitable for the destination and season.",
            ],
        },
    }

    key = "Iceland" if "iceland" in destination.lower() or "reykjavik" in destination.lower() else "default"
    data = local_tips_data[key]

    output = f"Local Expert Guide: {destination}\n"
    output += f"{'='*60}\n"

    output += "\n--- Customs & Etiquette ---\n"
    for i, tip in enumerate(data["customs"], 1):
        output += f"  {i}. {tip}\n"

    output += "\n--- Safety Notes ---\n"
    for i, tip in enumerate(data["safety"], 1):
        output += f"  {i}. {tip}\n"

    output += "\n--- Budget-Aware Local Tips ---\n"
    for i, tip in enumerate(data["money"], 1):
        output += f"  {i}. {tip}\n"

    output += "\n--- Packing Tips ---\n"
    for i, tip in enumerate(data["packing"], 1):
        output += f"  {i}. {tip}\n"

    return output


def get_hotel_location(destination: str) -> str:
    """Map a country-style destination to the main city used for hotel search."""
    if destination.lower() == "iceland":
        return "Reykjavik"
    if destination.lower() == "france":
        return "Paris"
    if destination.lower() == "japan":
        return "Tokyo"
    return destination


# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_flight_agent(destination: str, trip_dates: str):
    """Create the Flight Specialist agent."""
    return Agent(
        role="Flight Specialist",
        goal=f"Research and recommend the best flight options for the {destination} trip "
             f"({trip_dates}), prioritizing budget-friendly options first and direct flights whenever they remain cost-effective.",
        backstory="You are an experienced flight specialist with deep knowledge of "
                  "airline schedules, pricing patterns, and travel routes. You focus on budget airlines and cost savings above all, "
                  "but you still prefer direct flights when the price difference is reasonable. "
                  "You have booked thousands of flights and know the best times to fly. "
                  "You synthesize the curated research packet provided in the task into a clear recommendation.",
        verbose=True,
        allow_delegation=False
    )


def create_hotel_agent(destination: str, trip_dates: str):
    """Create the Accommodation Specialist agent."""
    hotel_location = get_hotel_location(destination)

    return Agent(
        role="Accommodation Specialist",
        goal=f"Suggest top-rated hotels in {hotel_location} for the {destination} trip "
             f"({trip_dates}), considering amenities, location, and value for money.",
        backstory="You are a seasoned accommodation expert with extensive knowledge of "
                  "hotels worldwide. You understand traveler needs and can match them with "
                  "perfect accommodations. You read reviews meticulously and know which "
                  "hotels offer the best experience for different budgets. You turn the provided hotel research packet "
                  "into concise, practical recommendations.",
        verbose=True,
        allow_delegation=False
    )


def create_itinerary_agent(destination: str, trip_duration: str):
    """Create the Travel Planner agent."""
    return Agent(
        role="Travel Planner",
        goal=f"Create a detailed day-by-day travel plan with activities and attractions "
             f"that maximize the {destination} experience in {trip_duration}.",
        backstory=f"You are a creative travel planner with a passion for {destination}. "
                  f"You have extensive knowledge of {destination}'s attractions, culture, and hidden gems. "
                  f"You create itineraries that are well-paced, exciting, and memorable. "
                  f"You consider travel times, weather, and traveler preferences to craft the perfect journey. "
                  f"You use the provided attractions packet to build a realistic, traveler-friendly schedule.",
        verbose=True,
        allow_delegation=False
    )


def create_budget_agent(destination: str):
    """Create the Financial Advisor agent."""
    return Agent(
        role="Financial Advisor",
        goal=f"Calculate total trip costs for {destination} and identify cost-saving opportunities "
             f"while maintaining quality. Synthesize the prior agent outputs into realistic budget scenarios.",
        backstory="You are a meticulous financial advisor specializing in travel budgeting. "
                  "You can analyze costs across flights, accommodations, activities, and meals. "
                  "You identify hidden costs and suggest smart ways to save money without "
                  "compromising the travel experience. You provide realistic budget estimates "
                  "by combining the concrete details from the earlier specialist agents.",
        verbose=True,
        allow_delegation=False
    )


def create_local_expert_agent(destination: str):
    """Create the Local Culture & Safety Specialist agent."""
    return Agent(
        role="Local Culture & Safety Specialist",
        goal=f"Provide practical local guidance for visiting {destination}, including customs, safety, packing, "
             f"and money-saving tips that improve the final plan.",
        backstory=f"You are a trusted local expert for travelers visiting {destination}. You know the local customs, "
                  f"seasonal safety concerns, transportation habits, and the small practical details that make trips "
                  f"smoother, safer, and more cost-effective. You always provide specific advice that other planners "
                  f"can use immediately.",
        verbose=True,
        allow_delegation=False
    )


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

def create_flight_task(flight_agent, destination: str, trip_dates: str, departure_city: str, research_packet: str):
    """Define the flight research task using a curated research packet."""
    return Task(
        description=f"Use the curated flight research packet below to compile a list of the best flight options "
                   f"from {departure_city} to {destination} for the trip ({trip_dates}). "
                   f"Do not call any tools or invent outside sources. Summarize 2-3 strong options from the packet, "
                   f"then recommend the best overall choice with clear reasoning about price, convenience, and baggage tradeoffs.\n\n"
                   f"FLIGHT RESEARCH PACKET\n{research_packet}",
        agent=flight_agent,
        expected_output=f"A detailed report with 2-3 flight options from {departure_city} to {destination} "
                       f"including airlines, times, duration, prices, and a recommendation grounded in the provided packet"
    )


def create_hotel_task(hotel_agent, destination: str, trip_dates: str, research_packet: str):
    """Define the hotel recommendation task using a curated research packet."""
    hotel_location = get_hotel_location(destination)

    return Task(
        description=f"Use the curated hotel research packet below to recommend the top 3-4 hotel options in "
                   f"{hotel_location} for the trip dates ({trip_dates}). Do not call any tools. Provide a mix of budget, "
                   f"mid-range, and luxury choices, then explain which option is best for a typical mid-range traveler.\n\n"
                   f"HOTEL RESEARCH PACKET\n{research_packet}",
        agent=hotel_agent,
        expected_output=f"A curated list of 3-4 hotel recommendations in {hotel_location} with details about amenities, "
                       f"ratings, nightly prices, and a best-fit recommendation grounded in the provided packet"
    )


def create_itinerary_task(itinerary_agent, destination: str, trip_duration: str, trip_dates: str, research_packet: str):
    """Define the itinerary planning task using a curated research packet."""
    return Task(
        description=f"Use the curated attractions packet below to create a detailed {trip_duration} itinerary for "
                   f"{destination} ({trip_dates}). Do not call any tools. Build a realistic day-by-day plan with a sensible pace, "
                   f"estimated timing, and a good mix of iconic sights and practical logistics.\n\n"
                   f"ATTRACTIONS PACKET\n{research_packet}",
        agent=itinerary_agent,
        expected_output=f"A detailed day-by-day itinerary for {destination} with realistic activities, estimated durations, "
                       f"pricing cues, and practical sequencing grounded in the provided packet"
    )


def create_budget_task(budget_agent, destination: str, trip_duration: str, research_packet: str):
    """Define the budget calculation task using prior outputs plus a cost packet."""
    return Task(
        description=f"Based on the REAL flight options, hotel recommendations, itinerary, and local expert guidance "
                   f"created by the other agents, calculate a comprehensive budget for the "
                   f"{trip_duration} {destination} trip using the concrete prices, recommendations, and local tips "
                   f"already gathered in the workflow. Include flights, accommodation, meals, "
                   f"activities, transportation, and miscellaneous expenses. Incorporate any local savings "
                   f"opportunities, cultural norms, or practical tips that affect spending. Provide total cost estimates "
                   f"for budget, mid-range, and luxury options based on the information already collected. Suggest "
                   f"genuine cost-saving tips based on current market conditions. Use the cost guide below as your reference packet "
                   f"for meals, transit, and daily spending assumptions. Do not call any tools.\n\n"
                   f"COST GUIDE PACKET\n{research_packet}",
        agent=budget_agent,
        expected_output=f"A comprehensive budget report with itemized costs for flights, "
                       f"accommodation, meals, activities with actual entry fees, transportation, "
                       f"and total realistic estimates at different budget levels, plus "
                       f"evidence-based cost-saving recommendations for a {trip_duration} trip to {destination}"
    )


def create_local_expert_task(local_expert_agent, destination: str, trip_dates: str, research_packet: str):
    """Define the local customs and safety task using a curated packet."""
    return Task(
        description=f"Provide destination-specific local guidance for {destination} during {trip_dates}. "
                   f"Include practical advice about customs, etiquette, transportation habits, weather-related "
                   f"safety, packing, and 4-6 tips that help travelers avoid unnecessary costs or mistakes. "
                   f"Make the guidance concrete enough that the BudgetAgent can use it in the final budget analysis. "
                   f"Use only the local guidance packet below.\n\n"
                   f"LOCAL GUIDANCE PACKET\n{research_packet}",
        agent=local_expert_agent,
        expected_output=f"A concise local guide for {destination} covering customs, safety, practical etiquette, "
                       f"packing advice, and money-saving local tips that can improve the overall travel plan"
    )


# ============================================================================
# CREW ORCHESTRATION
# ============================================================================

def main(destination: str = "Iceland", trip_duration: str = "5 days",
         trip_dates: str = "January 15-20, 2026", departure_city: str = "New York",
         travelers: int = 2, budget_preference: str = "mid-range"):
    """
    Main function to orchestrate the travel planning crew.

    Args:
        destination: Travel destination (e.g., "Iceland", "France", "Japan")
        trip_duration: Duration of trip (e.g., "5 days", "7 days")
        trip_dates: Specific dates (e.g., "January 15-20, 2026")
        departure_city: City you're departing from (e.g., "New York", "Los Angeles")
        travelers: Number of travelers
        budget_preference: Budget level ("budget", "mid-range", "luxury")
    """

    print("=" * 80)
    print("CrewAI Multi-Agent Travel Planning System (REAL API VERSION)")
    print(f"Planning a {trip_duration} Trip to {destination}")
    print("=" * 80)
    print()
    print(f"📍 Destination: {destination}")
    print(f"📅 Dates: {trip_dates}")
    print(f"✈️  Departure from: {departure_city}")
    print(f"👥 Travelers: {travelers}")
    print(f"💰 Budget: {budget_preference}")
    print()

    # Validate configuration before proceeding
    print("🔍 Validating configuration...")
    if not validate_config():
        print("❌ Configuration validation failed. Please set up your .env file.")
        exit(1)

    # Set environment variables for CrewAI (it reads from os.environ)
    # CrewAI uses OPENAI_API_KEY and OPENAI_API_BASE environment variables
    os.environ["OPENAI_API_KEY"] = Config.API_KEY
    os.environ["OPENAI_API_BASE"] = Config.API_BASE
    
    # For Groq compatibility, also set OPENAI_MODEL_NAME
    if Config.USE_GROQ:
        os.environ["OPENAI_MODEL_NAME"] = Config.OPENAI_MODEL

    print("✅ Configuration validated successfully!")
    print()
    Config.print_summary()
    print()
    print("⚠️  IMPORTANT: This version uses the configured LLM provider plus curated research packets")
    print("    Agents will synthesize provided travel data into recommendations")
    print()
    print("Tip: Check your API usage at https://platform.openai.com/account/usage")
    print()

    print("Preparing curated research packets...")
    flight_packet = search_flight_prices(destination, departure_city)
    hotel_packet = search_hotel_options(get_hotel_location(destination), trip_dates)
    itinerary_packet = search_attractions_activities(destination)
    local_packet = search_local_tips(destination)
    cost_packet = search_travel_costs(destination)
    print("Research packets prepared successfully!")
    print()

    # Create agents with destination parameters
    print("[1/5] Creating Flight Specialist Agent (researches real flights)...")
    flight_agent = create_flight_agent(destination, trip_dates)

    print("[2/5] Creating Accommodation Specialist Agent (researches real hotels)...")
    hotel_agent = create_hotel_agent(destination, trip_dates)

    print("[3/5] Creating Travel Planner Agent (researches real attractions)...")
    itinerary_agent = create_itinerary_agent(destination, trip_duration)

    print("[4/5] Creating Local Culture & Safety Specialist Agent...")
    local_expert_agent = create_local_expert_agent(destination)

    print("[5/5] Creating Financial Advisor Agent (analyzes real costs)...")
    budget_agent = create_budget_agent(destination)

    print("\n✅ All agents created successfully!")
    print()

    # Create tasks with destination parameters
    print("Creating tasks for the crew...")
    flight_task = create_flight_task(flight_agent, destination, trip_dates, departure_city, flight_packet)
    hotel_task = create_hotel_task(hotel_agent, destination, trip_dates, hotel_packet)
    itinerary_task = create_itinerary_task(itinerary_agent, destination, trip_duration, trip_dates, itinerary_packet)
    local_expert_task = create_local_expert_task(local_expert_agent, destination, trip_dates, local_packet)
    budget_task = create_budget_task(budget_agent, destination, trip_duration, cost_packet)

    print("Tasks created successfully!")
    print()

    # Create the crew with sequential task execution
    print("Forming the Travel Planning Crew...")
    print("Task Sequence: FlightAgent → HotelAgent → ItineraryAgent → LocalExpert → BudgetAgent")
    print()

    crew = Crew(
        agents=[flight_agent, hotel_agent, itinerary_agent, local_expert_agent, budget_agent],
        tasks=[flight_task, hotel_task, itinerary_task, local_expert_task, budget_task],
        verbose=True,
        process="sequential"  # Sequential task execution
    )

    # Execute the crew
    print("=" * 80)
    print("Starting Crew Execution...")
    print(f"Planning {trip_duration} trip to {destination} ({trip_dates})")
    print("=" * 80)
    print()

    try:
        result = crew.kickoff(inputs={
            "trip_destination": destination,
            "trip_duration": trip_duration,
            "trip_dates": trip_dates,
            "departure_city": departure_city,
            "travelers": travelers,
            "budget_preference": budget_preference
        })

        print()
        print("=" * 80)
        print("✅ Crew Execution Completed Successfully!")
        print("=" * 80)
        print()
        print(f"FINAL TRAVEL PLAN REPORT FOR {destination.upper()}:")
        print("-" * 80)
        print(result)
        print("-" * 80)

        # Save output to file
        output_filename = f"crewai_output_{destination.lower()}.txt"
        output_path = Path(__file__).parent / output_filename

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("CrewAI Multi-Agent Travel Planning System - Execution Report\n")
            f.write(f"Planning a {trip_duration} Trip to {destination}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Trip Details:\n")
            f.write(f"  Destination: {destination}\n")
            f.write(f"  Duration: {trip_duration}\n")
            f.write(f"  Dates: {trip_dates}\n")
            f.write(f"  Departure: {departure_city}\n")
            f.write(f"  Travelers: {travelers}\n")
            f.write(f"  Budget Preference: {budget_preference}\n\n")
            f.write(f"Execution Time: {datetime.now()}\n")
            f.write(f"API Version: CONFIGURED LLM PROVIDER ({'Groq-compatible' if Config.USE_GROQ else 'OpenAI-compatible'})\n")
            f.write("Data Source: Curated research packets + sequential agent synthesis\n\n")
            f.write("IMPORTANT NOTES:\n")
            f.write("- The plan is based on curated travel research packets embedded in the workflow\n")
            f.write("- Prices and availability should be verified before booking\n")
            f.write("- Weather conditions and attraction hours should be confirmed before travel\n\n")
            f.write("FINAL TRAVEL PLAN REPORT:\n")
            f.write("-" * 80 + "\n")
            f.write(str(result))
            f.write("\n" + "-" * 80 + "\n")

        print(f"\n✅ Output saved to {output_filename}")
        print("ℹ️  Note: This report is based on the configured LLM provider")
        print("    and curated travel research packets included in the workflow.")

    except Exception as e:
        print(f"\n❌ Error during crew execution: {str(e)}")
        print("\n🔍 Troubleshooting:")
        print("   1. Verify OPENAI_API_KEY is set: export OPENAI_API_KEY='sk-...'")
        print("   2. Check API key is valid and has sufficient credits")
        print("   3. Verify the installed CrewAI/OpenAI-compatible packages are working correctly")
        print("   4. Check your configured API provider status page")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow command line arguments to override defaults
    import sys

    kwargs = {
        "destination": "Iceland",
        "trip_duration": "5 days",
        "trip_dates": "January 15-20, 2026",
        "departure_city": "New York",
        "travelers": 2,
        "budget_preference": "mid-range"
    }

    # Parse command line arguments (optional)
    # Usage: python crewai_demo.py [destination] [duration] [departure_city]
    # Example: python crewai_demo.py "France" "7 days" "Los Angeles"
    if len(sys.argv) > 1:
        kwargs["destination"] = sys.argv[1]
    if len(sys.argv) > 2:
        kwargs["trip_duration"] = sys.argv[2]
    if len(sys.argv) > 3:
        kwargs["departure_city"] = sys.argv[3]
    if len(sys.argv) > 4:
        kwargs["trip_dates"] = sys.argv[4]
    if len(sys.argv) > 5:
        kwargs["travelers"] = int(sys.argv[5])
    if len(sys.argv) > 6:
        kwargs["budget_preference"] = sys.argv[6]

    main(**kwargs)
