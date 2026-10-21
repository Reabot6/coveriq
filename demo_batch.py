"""Fictional customers and catalogue entries for the batch demo."""


# These are opaque demo references, not hashes of the customers' names.
# The names and profile details below are fictional.
CUSTOMERS = [
    {
        "customer_id": "CUST-001",
        "name": "Chinedu",
        "customer_type": "salaried_customer",
        "stated_needs": ["long_term_savings"],
        "declared_assets": [],
        "consent_given": False,
    },
    {
        "customer_id": "CUST-002",
        "name": "Amaka",
        "customer_type": "marine_freight_operator",
        "stated_needs": ["vessel_protection"],
        "declared_assets": ["small_cargo_vessel"],
        "consent_given": False,
    },
]


# Product fit signals guide the next build. They are not underwriting rules
# or a claim that the customer qualifies for cover.
OFFERS = [
    {
        "offer_id": "PZL-SMART-SAVE",
        "name": "My Smart Save",
        "provider": "Prudential Zenith Life",
        "category": "savings_and_protection",
        "customer_need": "Build savings over time with life protection.",
        "fit_signals": [
            "stated long-term savings goal",
            "sustained positive cash flow",
        ],
        "message": "Would you like to review My Smart Save?",
        "catalogue_url": "https://prudentialzenith.com/",
    },
    {
        "offer_id": "PZL-EDUCATION-SAVINGS",
        "name": "Education savings plan",
        "provider": "Prudential Zenith Life",
        "category": "education_savings",
        "customer_need": "Save toward a child's education.",
        "fit_signals": ["stated education savings goal"],
        "message": "Would you like to review an education savings plan?",
        "catalogue_url": "https://prudentialzenith.com/",
    },
    {
        "offer_id": "ZGI-MOTOR",
        "name": "Motor Insurance",
        "provider": "Zenith General Insurance",
        "category": "motor_insurance",
        "customer_need": "Explore cover for a vehicle the customer owns or uses.",
        "fit_signals": ["declared vehicle ownership or use"],
        "message": "Would you like to review Zenith motor cover?",
        "catalogue_url": "https://www.zenithinsurance.com.ng/our-products",
    },
    {
        "offer_id": "ZGI-MARINE-HULL",
        "name": "Marine Hull",
        "provider": "Zenith General Insurance",
        "category": "marine_hull",
        "customer_need": "Explore cover for a vessel or watercraft.",
        "fit_signals": ["declared vessel ownership or operation"],
        "message": "Would you like to review Marine Hull cover?",
        "catalogue_url": "https://www.zenithinsurance.com.ng/our-products",
    },
    {
        "offer_id": "ZGI-MARINE-CARGO",
        "name": "Marine Cargo",
        "provider": "Zenith General Insurance",
        "category": "marine_cargo",
        "customer_need": "Explore cover for goods moved by sea.",
        "fit_signals": ["declared marine cargo or freight activity"],
        "message": "Would you like to review Marine Cargo cover?",
        "catalogue_url": "https://www.zenithinsurance.com.ng/our-products",
    },
]
