/* ClearPath Exam Check.
 * model.js (written by ClearPath_Decision_Tree.ipynb) holds the fitted StandardScaler and decision tree.
 * This file does the same two steps as the notebook pipeline in the browser: scale the inputs, then walk the tree.
 * The pure model code is exposed as window.ClearPathCore so it can be tested without a page. */
(function () {
  "use strict";
  var MODEL = window.CLEARPATH_MODEL;

  var LABELS = {
    study_hours: "Study hours per day",
    attendance: "Attendance (%)",
    sleep_hours: "Sleep hours per night",
    internet_usage: "Internet use",
    assignments_completed: "Assignments completed",
    previous_score: "Previous score"
  };
  var HINTS = {
    study_hours: "Dataset scale 1 to 11 (units to be confirmed)",
    attendance: "Dataset range 40 to 100 percent",
    sleep_hours: "Dataset range 4 to 9",
    internet_usage: "Dataset scale 1 to 11 (units to be confirmed)",
    assignments_completed: "Dataset range 0 to 20",
    previous_score: "Dataset range 35 to 95"
  };
  // Habits a student can act on. The tree moves in steps, so for each habit we look for the
  // smallest increase (in units of "step") that raises the estimate by at least MIN_GAIN.
  var LEVERS = [
    { key: "study_hours", step: 1, verb: "Study" , unit: "h per day" },
    { key: "assignments_completed", step: 1, verb: "Complete", unit: " assignments" },
    { key: "attendance", step: 5, verb: "Attend", unit: "% of classes" },
    { key: "sleep_hours", step: 1, verb: "Sleep", unit: "h per night" }
  ];
  var MIN_GAIN = 0.03;

  // Weakest-input advice. Only habits a student can change are ranked. "higher" says which direction is better.
  // Internet use is lower-is-better because it correlates negatively with exam score in the training data.
  var ADVICE = {
    study_hours: { higher: true, tip: "Try short, fixed study blocks (for example 2 x 45 minutes) on the same hours every day instead of one long session." },
    assignments_completed: { higher: true, tip: "Finish assignments in small pieces right after class and track them in a checklist so none slip through." },
    attendance: { higher: true, tip: "Treat every class as non-negotiable. If you must miss one, get the notes the same day." },
    sleep_hours: { higher: true, tip: "Keep a steady bedtime and wake time. Steady sleep before exams usually helps memory and focus." },
    internet_usage: { higher: false, tip: "Move social media and video apps out of your study blocks. Use a timer or app blocker." }
  };

  // 0 = worst value in the dataset range, 1 = best value, direction-aware.
  function strength(key, value) {
    var r = MODEL.ranges[key], span = r.max - r.min;
    var x = (value - r.min) / span;
    return ADVICE[key].higher ? x : 1 - x;
  }

  // Returns the weakest changeable input, a realistic target, and what the model says that target is worth.
  // Target = the class median (or one step better if already past it), so the advice stays reachable.
  function weakest(values) {
    var base = predict(values).onTrack, ranked = Object.keys(ADVICE).map(function (key) {
      return { key: key, value: values[indexOf(key)], strength: strength(key, values[indexOf(key)]) };
    }).sort(function (a, b) { return a.strength - b.strength; });
    var w = ranked[0], r = MODEL.ranges[w.key], dir = ADVICE[w.key].higher ? 1 : -1;
    var target = Math.round(r.median);
    if ((target - w.value) * dir <= 0) { target = w.value + dir; }
    target = Math.min(r.max, Math.max(r.min, target));
    var next = values.slice();
    next[indexOf(w.key)] = target;
    return {
      key: w.key, value: w.value, target: target, strength: w.strength,
      gain: predict(next).onTrack - base, atBest: target === w.value, tip: ADVICE[w.key].tip,
      higher: ADVICE[w.key].higher, runnerUp: ranked[1].key
    };
  }

  function indexOf(name) { return MODEL.features.indexOf(name); }

  /* StandardScaler, then DecisionTreeClassifier. scikit-learn compares float32 values, so Math.fround does too. */
  function predict(values) {
    var t = MODEL.tree, path = [], node = 0;
    var scaled = values.map(function (v, i) {
      return Math.fround((v - MODEL.scaler.mean[i]) / MODEL.scaler.scale[i]);
    });
    while (t.left[node] !== -1) {
      var f = t.feature[node];
      var goLeft = scaled[f] <= Math.fround(t.threshold[node]);
      path.push({
        feature: MODEL.features[f],
        value: values[f],
        cutoff: t.threshold[node] * MODEL.scaler.scale[f] + MODEL.scaler.mean[f],
        left: goLeft
      });
      node = goLeft ? t.left[node] : t.right[node];
    }
    var share = t.share[node];
    var onTrack = share[MODEL.classes.indexOf("On Track")];
    return {
      label: MODEL.classes[share.indexOf(Math.max.apply(null, share))],
      onTrack: onTrack, samples: t.samples[node], path: path
    };
  }

  function band(p) {
    if (p >= 0.8) return { name: "On track", cls: "good", text: "Habits like yours usually go with a 70+ exam score." };
    if (p >= 0.5) return { name: "Borderline", cls: "warn", text: "Close to the line. Small changes could tip this either way." };
    return { name: "Needs support", cls: "bad", text: "Habits like yours less often go with a 70+ score. Getting support early is a good idea." };
  }

  window.ClearPathCore = { predict: predict, band: band, weakest: weakest };
  if (typeof document === "undefined" || !MODEL) { return; }

  /* ---------- page ---------- */
  var fieldsBox = document.getElementById("fields");
  var form = document.getElementById("habit-form");
  var errors = document.getElementById("form-errors");
  var resultBody = document.getElementById("result-body");

  fieldsBox.innerHTML = MODEL.features.map(function (name) {
    var r = MODEL.ranges[name], start = Math.round(r.median);
    return '<div class="field"><div class="row"><label for="n_' + name + '">' + LABELS[name] + '</label>' +
      '<input class="num" type="number" inputmode="numeric" id="n_' + name + '" min="' + r.min + '" max="' + r.max + '" step="1" value="' + start + '"></div>' +
      '<input type="range" id="r_' + name + '" min="' + r.min + '" max="' + r.max + '" step="1" value="' + start + '" aria-label="' + LABELS[name] + ' slider">' +
      '<p class="hint">' + HINTS[name] + '</p></div>';
  }).join("");

  MODEL.features.forEach(function (name) {
    var num = document.getElementById("n_" + name), rng = document.getElementById("r_" + name);
    rng.addEventListener("input", function () { num.value = rng.value; num.removeAttribute("aria-invalid"); });
    num.addEventListener("input", function () { if (num.value !== "") { rng.value = num.value; } });
  });

  function readValues() {
    var values = [], problems = [];
    MODEL.features.forEach(function (name) {
      var el = document.getElementById("n_" + name), r = MODEL.ranges[name], v = Number(el.value);
      el.removeAttribute("aria-invalid");
      if (el.value.trim() === "" || !isFinite(v) || v % 1 !== 0 || v < r.min || v > r.max) {
        el.setAttribute("aria-invalid", "true");
        problems.push(LABELS[name] + " needs a whole number from " + r.min + " to " + r.max + ".");
      }
      values.push(v);
    });
    return { values: values, problems: problems };
  }

  function clampTo(name, v) { var r = MODEL.ranges[name]; return Math.min(r.max, Math.max(r.min, v)); }
  function pts(x) { return Math.round(x * 100); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function fmt(x) { return Number.isInteger(x) ? String(x) : x.toFixed(1); }

  function show(values) {
    var res = predict(values), p = res.onTrack, b = band(p);

    var all = values.slice();
    var levers = LEVERS.map(function (l) {
      var i = indexOf(l.key), r = MODEL.ranges[l.key];
      for (var v = values[i] + l.step; v <= r.max + l.step - 1; v += l.step) {
        var target = Math.min(r.max, v), next = values.slice();
        next[i] = target;
        var gain = predict(next).onTrack - p;
        if (gain >= MIN_GAIN) { all[i] = target; return { text: l.verb + " " + target + l.unit + " (now " + values[i] + ")", gain: gain, found: true }; }
        if (target === r.max) { break; }
      }
      return { text: l.verb + " more (" + LABELS[l.key].toLowerCase() + ")", gain: 0, found: false };
    }).sort(function (a, b) { return b.gain - a.gain; });
    var allP = predict(all).onTrack;

    var leverHtml = levers.map(function (l) {
      return "<li><span>" + l.text + "</span><span class=\"delta " + (l.found ? "up" : "same") + "\">" +
        (l.found ? "+" + pts(l.gain) + " pts" : "no gain found") + "</span></li>";
    }).join("");

    var pathHtml = res.path.map(function (s) {
      var cut = fmt(Math.round(s.cutoff * 100) / 100);
      return "<li><code>" + LABELS[s.feature] + "</code> is " + s.value + ", which is " + (s.left ? "at or below " : "above ") + cut + "</li>";
    }).join("");

    var wk = weakest(values), wkHtml;
    if (wk.atBest) {
      wkHtml = '<div class="block focus"><h3>Your focus area</h3><p>Your inputs are already at the best end of the dataset range. Keep your routine steady.</p></div>';
    } else {
      var dirWord = wk.higher ? "Raise" : "Lower";
      var gainTxt = wk.gain >= MIN_GAIN ? "The model estimates about <strong>+" + pts(wk.gain) + " pts</strong> from this change alone."
        : wk.gain > 0 ? "The model sees only a small gain (+" + pts(wk.gain) + " pts) from this change alone, but it is still a healthy habit."
        : "The model does not show a gain from this change alone, so combine it with the other changes above.";
      wkHtml = '<div class="block focus"><h3>Your focus area: ' + LABELS[wk.key].toLowerCase() + '</h3>' +
        '<p class="recommend"><strong>' + dirWord + ' ' + LABELS[wk.key].toLowerCase() + ' from ' + wk.value + ' to ' + wk.target + '.</strong> ' +
        'This is your weakest changeable input compared with the dataset range (' + pts(wk.strength) + '% of the way to the best value).</p>' +
        '<p class="sub">' + gainTxt + '</p><p class="sub">Tip: ' + wk.tip + '</p></div>';
    }

    resultBody.innerHTML =
      '<div class="big"><span class="pct">' + pts(p) + '%</span><span class="pill ' + b.cls + '">' + b.name + '</span></div>' +
      '<p class="sub">Estimated chance of scoring 70 or higher. ' + b.text + '</p>' +
      '<div class="bar" role="img" aria-label="Chance of 70 or higher: ' + pts(p) + ' percent"><i></i><i></i><i></i><span class="mark" style="left:' + Math.min(99, Math.max(1, p * 100)) + '%"></span></div>' +
      '<div class="scale"><span>0% needs support</span><span>50% borderline</span><span>80% on track</span></div>' +
      wkHtml + '<div class="block"><h3>Smallest changes that raise your estimate</h3><ul class="levers">' + leverHtml + '</ul>' +
      '<p class="sub">Smallest changes that help, together: <strong>' + pts(p) + '% to ' + pts(allP) + '%</strong>. These are patterns in the data, not promises.</p></div>' +
      '<div class="block"><h3>The tree\'s questions for you</h3><ol class="path">' + pathHtml + '</ol>' +
      '<p class="sub">The final group had ' + res.samples + ' training students; ' + pts(p) + '% of them scored 70+.</p></div>' +
      '<p class="note"><strong>Read this as a conversation starter.</strong> The model finds patterns in a sample dataset. It does not show that a habit causes a score.</p>';
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var r = readValues();
    if (r.problems.length) {
      errors.hidden = false;
      errors.innerHTML = r.problems.map(esc).join("<br>");
      resultBody.innerHTML = '<p class="placeholder">Fix the highlighted inputs, then check again.</p>';
      return;
    }
    errors.hidden = true;
    show(r.values);
  });

  document.getElementById("reset").addEventListener("click", function () {
    MODEL.features.forEach(function (name) {
      var m = Math.round(MODEL.ranges[name].median);
      document.getElementById("n_" + name).value = m;
      document.getElementById("r_" + name).value = m;
      document.getElementById("n_" + name).removeAttribute("aria-invalid");
    });
    errors.hidden = true;
    resultBody.innerHTML = '<p class="placeholder">Press <strong>Check my readiness</strong> to see your estimate.</p>';
  });

  var m = MODEL.metrics;
  document.getElementById("metrics").innerHTML =
    "<dt>Training accuracy</dt><dd>" + (m.trainingAccuracy * 100).toFixed(1) + "%</dd>" +
    "<dt>Validation accuracy</dt><dd>" + (m.validationAccuracy * 100).toFixed(1) + "%</dd>" +
    "<dt>Always-On-Track guess</dt><dd>" + (m.baseline * 100).toFixed(1) + "%</dd>" +
    "<dt>Needs Support caught</dt><dd>" + (m.recallNeedsSupport * 100).toFixed(0) + "%</dd>" +
    "<dt>Needs Support correct</dt><dd>" + (m.precisionNeedsSupport * 100).toFixed(0) + "%</dd>" +
    "<dt>Rows (train / validation)</dt><dd>" + m.trainRows + " / " + m.validRows + "</dd>";

  show(MODEL.features.map(function (n) { return Math.round(MODEL.ranges[n].median); }));
})();
