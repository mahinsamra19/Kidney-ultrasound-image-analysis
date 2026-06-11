import re

filepath = r"c:\Users\Welcome\Desktop\kidney_complete\renal-ai\frontend\templates\index.html"
with open(filepath, "r", encoding="utf-8") as f:
    html = f.read()

# 1. Update root variables
html = html.replace("--bg-deep: #050b14;", "--bg-deep: #f8fafc;")
html = html.replace("--glass-bg: rgba(16, 25, 43, 0.6);", "--glass-bg: rgba(255, 255, 255, 0.7);")
html = html.replace("--glass-border: rgba(255, 255, 255, 0.08);", "--glass-border: rgba(148, 163, 184, 0.3);")
html = html.replace("--glass-shine: rgba(255, 255, 255, 0.03);", "--glass-shine: rgba(255, 255, 255, 0.9);")
html = html.replace("--text-main: #f8fafc;", "--text-main: #0f172a;")
html = html.replace("--text-muted: #94a3b8;", "--text-muted: #475569;")
html = html.replace("--accent-base: #3b82f6;", "--accent-base: #2563eb;")
html = html.replace("--green: #10b981;", "--green: #059669;")
html = html.replace("--amber: #f59e0b;", "--amber: #d97706;")
html = html.replace("--red: #ef4444;", "--red: #dc2626;")

# Update background mesh
html = html.replace("background: var(--bg-deep); /* bg-mesh */", "background: linear-gradient(135deg, #f0f4f8, #e0e7ff);")
html = html.replace("rgba(59,130,246,0.15)", "rgba(59,130,246,0.08)")
html = html.replace("rgba(139,92,246,0.15)", "rgba(139,92,246,0.08)")
html = html.replace("rgba(16,185,129,0.1)", "rgba(16,185,129,0.05)")

# 2. Update box shadows and dark tints
html = html.replace("box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5)", "box-shadow: 0 20px 40px -12px rgba(0,0,0,0.05)")
html = html.replace("linear-gradient(to right, #fff, #94a3b8)", "linear-gradient(to right, #0f172a, #334155)")
html = html.replace("rgba(16, 25, 43, 0.8)", "rgba(241, 245, 249, 0.9)")
html = html.replace("border: 2px dashed rgba(255,255,255,0.15)", "border: 2px dashed rgba(15,23,42,0.15)")
html = html.replace("rgba(255,255,255,0.02)", "rgba(255,255,255,0.6)")
html = html.replace("rgba(255,255,255,0.05)", "rgba(15,23,42,0.05)")
html = html.replace("background: rgba(0,0,0,0.3)", "background: rgba(15,23,42,0.03)")
html = html.replace("rgba(255,255,255,0.1)", "rgba(15,23,42,0.08)")

# Keep image dark background proper
html = html.replace("background: #000;", "background: #1e293b;") 

# 3. Update text colors (like #fff)
html = html.replace("color: #fff;", "color: var(--text-main);")
html = html.replace("color:white;", "color:var(--text-main);")
html = html.replace("color:#cbd5e1;", "color:var(--text-main);")
html = html.replace("background: rgba(255,255,255,0.03)", "background: rgba(255,255,255,0.7)")
html = html.replace("color: #e2e8f0;", "color: var(--text-main);")
html = html.replace("color: #cbd5e1;", "color: var(--text-main);")

# Update probability text
html = html.replace("'#fff'", "'var(--text-main)'")

# 4. Remove probabilities HTML block and replace with the biomarkers
prob_block = """    <div class="grid-cols-2">
      <!-- Probabilities -->
      <div class="glass-panel" style="padding: 1.5rem;">
        <div class="block-title">Pathology Probabilities</div>
        <div id="prob-bars"></div>
      </div>
      
      <!-- Key Features -->
      <div class="glass-panel" style="padding: 1.5rem;">
        <div class="block-title">Identified Biomarkers</div>
        <div class="feature-list" id="features-el"></div>
      </div>
    </div>"""

new_feat_block = """    <div class="glass-panel" style="padding: 1.5rem; margin-bottom: 1.5rem;">
      <div class="block-title" style="margin-bottom: 1.5rem;">Identified Biomarkers & Confirmation Probabilities</div>
      <div class="feature-list" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem;" id="features-el"></div>
    </div>"""
html = html.replace(prob_block, new_feat_block)

# 5. Update JS render
js_feat = """  document.getElementById('features-el').innerHTML = feats.map(f => `
    <div class="feat-item">
      <div class="feat-dot" style="color:${c.main}; background:${c.main}"></div>
      <div class="feat-text">
        <h5>${f.name}</h5>
        <p>${f.desc}</p>
      </div>
    </div>
  `).join('');"""

js_new_feat = """  document.getElementById('features-el').innerHTML = feats.map(f => `
    <div class="feat-item" style="flex-direction: column; gap: 8px;">
      <div style="display: flex; align-items: start; gap: 10px;">
          <div class="feat-dot" style="color:${c.main}; background:${c.main}; flex-shrink: 0; margin-top: 8px;"></div>
          <div style="flex: 1;">
              <div style="display: flex; justify-content: space-between; margin-bottom: 4px; align-items: center;">
                  <h5 style="margin: 0; font-size: 15px; font-weight: 600; color: var(--text-main);">${f.name}</h5>
                  <span style="font-size: 13px; font-weight: 700; color: ${c.main};">${f.prob}%</span>
              </div>
              <div style="height: 6px; background: rgba(15,23,42,0.06); border-radius: 3px; overflow: hidden; margin-bottom: 10px;">
                  <div style="height: 100%; width: ${f.prob}%; background: ${c.main}; border-radius: 3px;"></div>
              </div>
              <p style="font-size: 13.5px; line-height: 1.6; color: var(--text-muted); margin: 0;">${f.desc}</p>
          </div>
      </div>
    </div>
  `).join('');"""
html = html.replace(js_feat, js_new_feat)

# Remove JS prob bars update
js_prob = """  // Probabilities
  const pContainer = document.getElementById('prob-bars');
  let probs = d.all_probs;
  if (!probs) probs = {}; 
  const sorted = Object.entries(probs).sort((a,b)=>b[1]-a[1]);
  pContainer.innerHTML = sorted.map(([cls, pct]) => {
    const isLead = cls === d.diagnosis;
    const barColor = isLead ? c.main : 'rgba(255,255,255,0.1)';
    return `<div class="prob-row">
      <div class="prob-lbl" style="color: ${isLead ? '#fff' : 'var(--text-muted)'}">${capitalize(cls)}</div>
      <div class="prob-track"><div class="prob-fill" style="width:${pct}%; background:${barColor}; box-shadow:${isLead?'0 0 10px '+c.main:'none'};"></div></div>
      <div class="prob-val">${pct.toFixed(1)}%</div>
    </div>`;
  }).join('');"""
html = html.replace(js_prob, "// Probabilities block removed")


with open(filepath, "w", encoding="utf-8") as f:
    f.write(html)

print("Theme updated successfully.")
