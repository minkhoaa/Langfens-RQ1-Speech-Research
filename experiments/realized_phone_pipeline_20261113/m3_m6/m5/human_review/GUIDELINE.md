# Human review batch — labels are empty

100 real TRAIN-speaker utterances:50 teacher-disagreement/low-confidence
selection plus50 seeded random controls. Original audio copied unchanged,
checksums verified; no new speakers and no human labels yet. Models' phone
hypotheses are deliberately absent from the review form to reduce anchoring.

Independently annotate actual emitted phones before comparing with the supplied
prompt/reference. Preserve substitutions, omissions, each exact insertion gap
and multiple inserted phones. Mark uncertain/deviation/intermediate sounds and
alternative plausible realizations; do not force them into a guessed phone.
Define acceptable accent variants, linking/reduction, silence and hesitation
consistently. Have two independent annotators and adjudicate disagreements;
retain both original versions. Fill provenance and annotator IDs. Gold
timestamps, if collected for audit, must never become inference inputs.

The existing dictionary prompt profile can differ from corpus pronunciations;
do not use an automatic lexicon mismatch as proof of a learner error. Empty
templates are not human gold, not an IAA study and not fresh-speaker confirmation.
Dataset CC-BY-NC4.0 research provenance applies. Do not publish these audio files
or use them as a sealed test. Human verification and speaker expansion remain
outstanding; the agent has not supplied annotator judgments.
