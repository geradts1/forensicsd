# Ethics, intended use, and scope

## What this toolkit is for

`prnucf` exists to **measure and defend** the reliability of PRNU
source-camera attribution. Concretely, it lets a forensic practitioner:

1. run the published fingerprint-copy attack against *their own* attribution
   pipeline, to find the conditions under which a PCE-only conclusion can be
   fooled; and
2. detect that attack with the **triangle test**, and calibrate its false-
   positive rate against a reference cohort.

The attack and the defence ship together on purpose. An attack without its
detector teaches an examiner nothing about robustness; the detector is the
point of the package.

## What it is *not* for

Using PRNU injection to make a questioned image attribute to a camera that
did not take it — i.e. to **fabricate source-camera evidence against a
person** — is evidence tampering and, in most jurisdictions, a crime. This
toolkit must not be used for that, and its design deliberately avoids making
it convenient:

- It operates on fingerprints and images **you own or have consent to use**.
  There is no "harvest a target's public photos" pipeline, and none should be
  added.
- The injection implements only the **already-published additive model**. It
  contains **no refinements aimed at defeating the triangle test** or any
  specific deployed attribution system. The `match_target_pce` helper exists
  so an evaluator can find the strength needed to cross a decision threshold —
  the same operation reveals that the triangle test still fires at that
  strength.
- The removal/anonymisation functions are included for completeness of the
  robustness evaluation (attribution-denial scenarios), not as an evasion aid.

## Why publishing the attack is defensible here

The fingerprint-copy attack has been in the peer-reviewed literature since
2011 (Goljan, Fridrich & Chen). Withholding a clean reference implementation
does not deny the capability to a motivated adversary; it only denies
examiners a convenient way to test whether their own conclusions are robust
to it. The responsible posture — and the one this package takes — is to pair
the attack with its detector and to frame the whole thing as a
counter-forensics *evaluation* tool.

## If you extend this

- Do not add target-acquisition tooling (scrapers, "paste into this photo of
  a suspect" flows).
- Do not add evasion refinements whose only purpose is to beat the triangle
  test or a named real system. If your research requires studying an improved
  attack, study the improved **detector** in the same work and disclose
  responsibly.
- Keep the own-data / consented-data constraint.

## Casework caution

A triangle-test result is one line of evidence. A flag indicates a questioned
image shares a fingerprint statistically tied to a specific image pool; it is
not, by itself, proof of intent or identity. Interpret it alongside
provenance, the processing chain, and the rest of the case material, and
report uncertainty honestly — the same standard that applies to the PCE
conclusion it is protecting.
