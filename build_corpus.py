"""Generate the original, synthetic hospitality shift notes shipped with this project."""

from pathlib import Path


VENUES = {
    "train": [
        ("Alder House", "library", "garden", "blue ledger"),
        ("Birch Hotel", "lounge", "courtyard", "desk diary"),
        ("Copper Inn", "snug", "terrace", "green binder"),
        ("Dune Lodge", "sunroom", "patio", "shift book"),
        ("Elm Rooms", "gallery", "orchard", "desk notebook"),
        ("Fern Hall", "parlour", "veranda", "red folder"),
        ("Grove Hotel", "atrium", "roof deck", "front desk log"),
        ("Hazel House", "dining room", "side garden", "guest register"),
        ("Ivory Inn", "tea room", "river deck", "paper planner"),
        ("Juniper Lodge", "reading room", "back lawn", "handover book"),
        ("Kestrel Hotel", "club room", "balcony", "grey binder"),
        ("Linden Hall", "drawing room", "front lawn", "desk register"),
        ("Maple Rooms", "breakfast room", "stone yard", "shift planner"),
        ("Nettle House", "studio", "greenhouse", "white notebook"),
        ("Olive Inn", "music room", "south lawn", "yellow folder"),
        ("Pine Lodge", "winter room", "pool deck", "black ledger"),
    ],
    "validation": [
        ("Quarry Hotel", "coffee room", "pond deck", "silver binder"),
        ("Rowan Hall", "sitting room", "herb garden", "night diary"),
        ("Spruce House", "map room", "west lawn", "linen notebook"),
    ],
    "test": [
        ("Thistle Lodge", "bay room", "hill terrace", "violet folder"),
        ("Umber Inn", "piano room", "east garden", "brown ledger"),
        ("Willow Hotel", "river room", "apple yard", "morning diary"),
    ],
}

CASES = [
    (
        "Booking request",
        "A guest asked for a table for {size} in the {inside} at {time}.",
        "Waiting for the floor manager to check the table plan.",
        "Call the guest when the table and time are confirmed.",
    ),
    (
        "Private hire",
        "A caller asked to hire the {outside} for a group of {size}.",
        "Date and guest count are in the {log}.",
        "Ask the events lead to price the hire and reply after {time}.",
    ),
    (
        "Lost property",
        "A guest left a {colour} coat in the {inside} after dinner.",
        "The desk has a description and a callback number.",
        "Check the cloakroom and update the {log} before {time}.",
    ),
    (
        "Access enquiry",
        "A caller asked about step-free access to the {outside}.",
        "The desk has the party size of {size} and the arrival time.",
        "Check the route with the duty manager before calling back.",
    ),
    (
        "Weather change",
        "Rain forced a party of {size} to leave the {outside}.",
        "The team offered seats in the {inside} at {time}.",
        "Record the move in the {log} and brief the next shift.",
    ),
    (
        "Noise complaint",
        "A guest reported noise near the {inside} just after {time}.",
        "The duty manager has the room number and callback details.",
        "Check the area and call the guest with an update.",
    ),
    (
        "Dietary enquiry",
        "A guest asked about dietary options for a party of {size}.",
        "The kitchen has the question and is checking the menu.",
        "Call back after the chef replies and update the {log}.",
    ),
    (
        "Supplier arrival",
        "A driver plans to deliver {size} boxes near the {outside} at {time}.",
        "The order number is in the {log}.",
        "Meet the driver at the service door and count the boxes.",
    ),
    (
        "Payment query",
        "A caller asked for a receipt for an event in the {inside}.",
        "The desk noted the event time and invoice reference.",
        "Ask accounts to verify the payment before sending the receipt.",
    ),
    (
        "Staff handover",
        "A party of {size} is due in the {inside} at {time}.",
        "The early shift left the table plan in the {log}.",
        "Check the room setup with the floor team before doors open.",
    ),
]

SHIFTS = [
    ("Early", "10:30", 4, "navy"),
    ("Lunch", "12:15", 6, "grey"),
    ("Late", "16:45", 8, "green"),
    ("Evening", "19:00", 12, "black"),
]


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    for split, venues in VENUES.items():
        notes = []
        for venue, inside, outside, log in venues:
            for shift, time, size, colour in SHIFTS:
                for topic, detail, status, next_step in CASES:
                    fields = dict(
                        inside=inside, outside=outside, log=log,
                        time=time, size=size, colour=colour,
                    )
                    notes.append(
                        f"Venue: {venue}\n"
                        f"Shift: {shift}\n"
                        f"Topic: {topic}\n"
                        f"Detail: {detail.format(**fields)}\n"
                        f"Status: {status.format(**fields)}\n"
                        f"Next step: {next_step.format(**fields)}\n"
                    )
        with (data_dir / f"{split}.txt").open("w", encoding="utf-8", newline="\n") as output:
            output.write("\n".join(notes))


if __name__ == "__main__":
    main()
