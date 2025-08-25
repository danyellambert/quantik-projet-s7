# 🎯 Quantik AI Project

This repository contains the Quantik game implementation, a graphical interface, and multiple AI players for the S7 project.

---

## 📂 Project Structure

```
quantik/
├── ai_players/        # All AI implementations (one folder per participant)
│   ├── template/      # Base template for creating a new AI
│   ├── student1/         # Example AI by student1
│   ├── student2/           # Example AI by student2
├── core/              # Core game logic (rules, data structures)
├── gui/               # Graphical interface
├── tournament/        # Tournament scripts for AI vs AI evaluation
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── .gitignore         # Files and folders to ignore in Git
```

---

## 👥 Instructions for Team Members

### 1️⃣ Copy the Template AI
1. Navigate to:
   ```
   ai_players/template/
   ```
2. Copy `algorithme.py` into a **new folder named after you**:
   ```
   ai_players/<your_name>/
   ```
3. Edit the code to implement your own strategy.

---

### 2️⃣ Test Your AI in the Interface
Run the game with:
```bash
python -m gui.app
```
Then select your AI in the dropdown menu and play against it.

---

### 3️⃣ Commit Your Changes
Before pushing, always **pull** the latest version:
```bash
git pull
git add .
git commit -m "Your message here"
git push
```

---

### 4️⃣ Run the Tournament
1. Place all AIs inside `ai_players/` (each in its own folder).
2. Run:
   ```bash
   python -m tournament.runner
   ```
3. View the results: win rates, performance stats, and more.

---

## 📌 Tips
- Keep your AI **compatible** with the template interface (`get_move()` method).
- Comment your code in **French** as required for the project.
- Test against different opponents to improve robustness.

---

## 🛠 Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
