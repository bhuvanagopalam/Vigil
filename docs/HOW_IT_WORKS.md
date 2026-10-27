# How Vigil measures it

## Part 1: the pulse (photoplethysmography, PPG)

### What the phone captures

- The rear camera runs with the flash on, and your fingertip covers both the lens and the flash. The flash light enters the finger, scatters through the tissue, and some of it comes back out into the lens. The whole frame is just a red glow.
- For every frame we shrink the image to 64 by 48 pixels and take the average red, green and blue value. That gives one number per colour per frame, with a timestamp. 20 seconds at 30 to 60 frames a second is 600 to 1,200 samples.
- Only those averages go to the laptop, never the image.

### Why the red value moves with the heartbeat

- Blood absorbs light. The more haemoglobin in the light path, the less light comes back to the lens. This is the Beer-Lambert law, the same physics as a hospital finger clip.
- Each heartbeat pushes a small extra volume of arterial blood into the fingertip. For a fraction of a second there is more haemoglobin in the path, more light is absorbed, and the red value dips by about 1 to 2 percent. Between beats it recovers.
- Red is the channel we use because red light penetrates tissue deepest and, with the flash on, green and blue are nearly black. The code checks this: if the mean red is above 120 and green below 60, the finger is on properly and we use red.

### How we turn that into numbers

- Drop the first 2 seconds, because the camera is still settling its exposure.
- Flip the sign, so a beat becomes a peak instead of a dip.
- Subtract a 1-second moving average. That removes slow drift from finger pressure and exposure, leaving only the fast pulse.
- Band-pass filter 0.7 to 3 Hz, which is 42 to 180 beats a minute. Everything slower or faster is noise.
- Take the spectrum. The strongest frequency in that band is the heart rate. The ratio of that peak to the rest of the spectrum is our signal-to-noise figure.
- Then find individual beats, guided by that estimate. Each pulse has a small second bump after the main peak, the dicrotic notch, which naive peak finders count as a second beat. We refuse any peak closer than 0.6 of the expected beat interval and keep only intervals within 25 percent of it. If we get at least 5 clean beats, heart rate comes from the median beat interval. Otherwise the spectral estimate stands.
- **HRV** is the variability of those beat intervals: the root mean square of successive differences, the standard measure called RMSSD. Needs clean beats, which is why a moving finger blanks it.
- **Perfusion index** is the height of the pulse wave divided by the steady red level. How much of the signal is pulsing blood.
- **Breathing** rides on the pulse in three ways: each breath changes the height of the beats, the spacing of the beats, and the baseline level. We extract all three, resample them, filter 0.15 to 0.6 Hz, which is 9 to 36 breaths a minute, and take the spectral peak.
- A quality score combines signal-to-noise and beat regularity. Under 25 percent the reading is marked unreadable.

## Part 2: capillary refill

- Same camera, same red average, but now you press the finger hard on the lens for 4 seconds, then relax without lifting for 6.
- Pressing squeezes blood out of the capillaries. Less haemoglobin in the path, less absorption, the red value rises. On the real trace it went from about 185 to 199.
- When you relax, blood floods back in. The red value falls sharply, overshoots a little as the vessels open wider than before, then settles. On the real trace it fell to 166 within a second and settled at 175.
- The code smooths the trace, takes its slope, and finds the steepest fall near the moment the screen said "relax", allowing up to 3.5 seconds for reaction time. The fall has to be steeper than 8 units a second and the total drop at least 12 units, otherwise it refuses to report a number. Without that rule, a slow drift with no real press was once read as a 4-second refill.
- The blanched level is the top of the trace just before the fall. The baseline is the level in the last 2 seconds, after the overshoot. The onset is the last moment the trace was still at the blanched level.
- **Refill time** is the time from that onset until the trace has covered 90 percent of the way back to baseline. Real reading: 0.70 seconds.
- Clinically, under 2 seconds is normal. Over 2 is poor peripheral circulation, and over 3 is strongly linked to shock and sepsis mortality. A nurse does this by pressing a fingernail and counting in whole seconds. The camera resolves it to a frame, about 30 milliseconds.

## Part 3: the room camera

- A second phone on a stand watches the waiting room. The nurse or teammate taps each chest once and types the chair number written on the chair.
- For each box we track the vertical shift of the chest's brightness profile against the frame half a second earlier, with sub-pixel precision. Sideways sway is rejected. The shifts are integrated into a breathing waveform.
- The breathing rate is the spectral peak between 0.1 and 0.7 Hz, 6 to 42 breaths a minute, over the last 24 seconds. If the movement collapses for about 8 seconds the seat is flagged as apnea.
- The patient types the same chair number at check-in, so that seat's breathing is tied to their row on the board and feeds their score between finger checks.

## Part 4: score and risk

- **Score** follows NEWS2 bands for heart rate and breathing rate, the UK early-warning score, plus: 1 point for a 20 percent rise in heart rate or 4 more breaths since arrival, 2 for capillary refill over 2 seconds, 1 for weak perfusion, 2 for a red-flag complaint, 1 for pain of 8 or more, and camera findings: 3 for no breathing movement, NEWS points for fast camera breathing. Every point comes with its reason on the board. The score decides the order.
- **Risk** is a logistic model trained on 3,964 ICU patients from the PhysioNet/CinC 2019 sepsis challenge, 151,000 hourly rows: heart rate, breathing (the live camera figure when higher), age, change over 3 hours and since arrival, predicting sepsis within 6 hours. Test AUROC 0.67 against 0.64 for the NEWS bands alone. Evidence the model never saw raises it by fixed log-odds and is listed under the number: refill over 2 s, HRV under 20 ms, weak perfusion, camera apnea, a red-flag complaint. It is shown as a percentile of the training hours. It informs, it does not sort.

## What it is not

A hospital clip uses two wavelengths, red and infrared, and their ratio gives oxygen saturation. We have one, so no SpO2, and no blood pressure. It is a reassessment aid for a nurse, not a diagnostic.
