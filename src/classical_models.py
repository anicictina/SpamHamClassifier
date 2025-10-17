from typing import Dict, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, make_scorer
from .preprocess import preprocess_series, build_vectorizer
from .evaluate import compute_metrics, save_confusion_matrix, save_roc_curve

# F1 scorer sa pozitivnom klasom "spam"
F1_SPAM = make_scorer(f1_score, pos_label="spam")

def train_classical(
    texts,
    labels,
    features: str = "tfidf",
    test_size: float = 0.2,
    random_state: int = 42,
    outdir: str = "outputs",
) -> Tuple[Dict, object]:
    # 1) čišćenje teksta i normalizacija labela
    X = preprocess_series(texts)
    y = labels.astype(str).str.lower()

    # 2) stratifikovana podela
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3) vektorizacija
    vectorizer = build_vectorizer(features)

    # 4) kandidati
    candidates = {
        "NaiveBayes": Pipeline([("vec", vectorizer), ("clf", MultinomialNB())]),
        "LogReg": Pipeline([
            ("vec", vectorizer),
            ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
        ]),
        "LinearSVM": Pipeline([
            ("vec", vectorizer),
            ("clf", LinearSVC(class_weight="balanced")),
        ]),
    }

    # 5) mreže hiperparametara
    param_grid = {
        "NaiveBayes": {"clf__alpha": [0.5, 1.0]},
        "LogReg": {"clf__C": [0.5, 1.0, 2.0]},
        "LinearSVM": {"clf__C": [0.5, 1.0, 2.0]},
    }

    results: Dict = {}
    best_name, best_score, best_model = None, -1.0, None

    # 6) GridSearch + evaluacija
    for name, pipe in candidates.items():
        grid = GridSearchCV(
            pipe,
            param_grid[name],
            cv=3,
            n_jobs=-1,
            scoring=F1_SPAM,  # ključna promena
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        # predikcije
        y_pred = model.predict(X_test)

        # skorovi za ROC (proba ili decision_function)
        y_proba = None
        try:
            clf = model.named_steps["clf"]
            if hasattr(clf, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None

        # metrike
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # vizuali
        from .utils import ensure_dir
        from os.path import join
        ensure_dir(outdir)
        save_confusion_matrix(y_test, y_pred, join(outdir, "confusion_matrices"), name)
        if y_proba is not None:
            save_roc_curve(y_test, y_proba, join(outdir, "roc_curves"), name)

        results[name] = {"best_params": grid.best_params_, **metrics}

        if metrics.get("f1", -1.0) > best_score:
            best_name, best_score, best_model = name, metrics["f1"], model

    report = {
        "results": results,
        "best_model_name": best_name,
        "best_f1": best_score,
    }
    return report, best_model
