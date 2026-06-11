import sys

path = r'c:\Users\Welcome\Desktop\kidney_complete\renal-ai\backend\models\inference.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find exact line numbers we need to replace
load_line = None
tensor_line = None
m1_run_line = None
m2_run_line = None
gradcam_line = None

for i, line in enumerate(lines):
    stripped = line.strip()
    if 'input_tensor = preprocess(pil_img)' in stripped:
        tensor_line = i
    if 'logits1 = registry.model1(input_tensor)' in stripped:
        m1_run_line = i
    if 'logits2 = registry.model2(input_tensor)' in stripped:
        m2_run_line = i
    if 'logits = model(input_tensor)' in stripped and m2_run_line and i > m2_run_line:
        # Could be two instances - capture both
        pass
    if 'input_for_gcam = preprocess(pil_img)' in stripped:
        gradcam_line = i

sys.stdout.buffer.write(f"tensor_line={tensor_line}, m1_run={m1_run_line}, m2_run={m2_run_line}, gradcam={gradcam_line}\n".encode('utf-8'))

# Fix 1: input_tensor line -> two separate tensors
if tensor_line is not None:
    indent = '    '
    lines[tensor_line] = (
        indent + "# Each model has its own preprocessing (different normalization)\n"
        + indent + "input_tensor_m1 = registry.preprocess1(pil_img).unsqueeze(0).to(registry.device)\n"
        + indent + "input_tensor_m2 = registry.preprocess2(pil_img).unsqueeze(0).to(registry.device)\n"
    )
    sys.stdout.buffer.write(b"Fix 1 applied: split input tensors\n")

# Fix 2: M1 uses input_tensor_m1
if m1_run_line is not None:
    lines[m1_run_line] = lines[m1_run_line].replace('registry.model1(input_tensor)', 'registry.model1(input_tensor_m1)')
    sys.stdout.buffer.write(b"Fix 2 applied: M1 uses input_tensor_m1\n")

# Fix 3: M2 uses input_tensor_m2
if m2_run_line is not None:
    lines[m2_run_line] = lines[m2_run_line].replace('registry.model2(input_tensor)', 'registry.model2(input_tensor_m2)')
    sys.stdout.buffer.write(b"Fix 3 applied: M2 uses input_tensor_m2\n")

# Fix 4: standalone m2 logits line (search again separately)
for i, line in enumerate(lines):
    if 'logits = model(input_tensor)' in line:
        # Determine which block it's in by looking for context
        lines[i] = line.replace(
            'logits = model(input_tensor)',
            'logits = model(input_tensor_m2 if model is registry.model2 else input_tensor_m1)'
        )
        sys.stdout.buffer.write(f"Fix 4 applied at line {i+1}\n".encode('utf-8'))

# Fix 5: GradCAM preprocessing
if gradcam_line is not None:
    indent = '    '
    lines[gradcam_line] = (
        indent + "# Use correct preprocessing for Grad-CAM matching the selected model\n"
        + indent + "if gcam is registry.gcam2:\n"
        + indent + "    input_for_gcam = registry.preprocess2(pil_img).unsqueeze(0).to(registry.device)\n"
        + indent + "else:\n"
        + indent + "    input_for_gcam = registry.preprocess1(pil_img).unsqueeze(0).to(registry.device)\n"
    )
    sys.stdout.buffer.write(b"Fix 5 applied: GradCAM preprocessing\n")

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

sys.stdout.buffer.write(b"All patches applied successfully!\n")
