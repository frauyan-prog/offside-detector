import ast
from collections import Counter

tree = ast.parse(open('offside_detector/pitch_keypoint_detector.py', encoding='utf-8').read())

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'PitchKeypointDetector':
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        
        targets = ['calibrate', 'calibrate_multiframe', 'voter_homography', 
                   'validate_homography', 'calibration_quality', 'draw_pitch_overlay']
        for m in targets:
            status = "OK" if m in methods else "MISSING"
            print(f"  {m}: {status}")
        
        counts = Counter(methods)
        dups = [k for k, v in counts.items() if v > 1]
        print(f"  Duplicates: {dups if dups else 'None'}")
        print(f"  Total methods: {len(methods)}")
