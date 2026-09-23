-- A card is a number on a piece of paper that belongs to one child. It is
-- not a biometric, which is the point of it: a school that will not consent
-- to storing faces can still have attribution and per-child marks.
--
-- A table rather than a column on the child, because cards are lost,
-- swapped and re-issued, and the old one has to stay readable.
CREATE TABLE IF NOT EXISTS cards (
    id           TEXT PRIMARY KEY,
    org_id       TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    student_id   TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    marker_id    INTEGER NOT NULL,
    issued_at    REAL NOT NULL,
    withdrawn_at REAL
);
CREATE INDEX IF NOT EXISTS ix_cards_student ON cards(org_id, student_id);

-- One live card per marker: handing the same number to two children is how
-- one child's answers end up against another's name.
CREATE UNIQUE INDEX IF NOT EXISTS ux_cards_marker ON cards(org_id, marker_id)
    WHERE withdrawn_at IS NULL;
