Feuille de route
1. Initialisation

    Créer le dépôt Git et l’arborescence du projet.

    Mettre en place un environnement virtuel venv ou conda.

    Créer requirements.txt ou environment.yml.

    Vérifier les imports de base : faiss, langchain, embeddings, client Mistral.

    Rédiger un README.md minimal avec installation, lancement et structure du projet.

Livrables :

    environnement reproductible ;

    fichier de dépendances ;

    README de démarrage.
    Faiss existe en paquet faiss-cpu, ce qui est cohérent avec la consigne de portabilité, et FastAPI fournit une mise en route très simple pour une API locale.

2. Collecte des données

    Choisir un périmètre simple : une ville ou une région.

    Récupérer les événements OpenAgenda/Opendatasoft.

    Filtrer par localisation et par période : historique d’un an + événements à venir.

    Nettoyer les champs utiles : titre, description, date, lieu, ville, catégorie.

    Sauvegarder un fichier propre prêt à être indexé.

Livrables :

    script fetch_events.py ;

    dataset nettoyé, par exemple JSON ou CSV ;

    premiers tests unitaires sur le filtrage.
    L’idée est surtout d’obtenir un jeu de données stable, cohérent et pas trop gros pour rester simple à manipuler dans un POC scolaire.

3. Préparation RAG

    Transformer chaque événement en document texte structuré.

    Découper les documents en chunks.

    Générer les embeddings.

    Construire l’index vectoriel avec FAISS.

    Conserver aussi les métadonnées utiles pour pouvoir les renvoyer dans la réponse.

Livrables :

    script build_index.py ;

    index FAISS sauvegardé ;

    test simple de recherche sémantique.
    LangChain s’interface avec FAISS pour la recherche vectorielle, ce qui colle bien au niveau attendu pour un POC académique.

4. Chatbot RAG

    Créer une fonction centrale ask(question) ou une classe RAGService.

    Faire : recherche des chunks pertinents → construction du prompt → appel Mistral → génération de réponse.

    Retourner aussi les sources utilisées.

    Ne pas gérer l’historique conversationnel, puisque le sujet dit qu’il n’est pas nécessaire.

Livrables :

    pipeline RAG fonctionnel ;

    quelques exemples de questions/réponses ;

    script de test manuel.
    Le but n’est pas un assistant conversationnel complexe, mais une réponse augmentée correctement fondée sur les événements indexés.

5. API REST

    Créer l’API avec FastAPI.

    Ajouter POST /ask pour poser une question.

    Ajouter POST /rebuild ou GET /rebuild pour reconstruire l’index.

    Ajouter GET /health pour vérifier que le service fonctionne.

    Tester l’API avec requests ou httpx.

Livrables :

    API locale ;

    Swagger auto-généré sur /docs ;

    fichier api_test.py.
    FastAPI génère automatiquement la documentation interactive et facilite les tests rapides, ce qui est pratique pour une soutenance.

Qualité
6. Évaluation

    Construire un petit jeu de test annoté, par exemple 20 à 50 questions.

    Prévoir pour chaque question une réponse de référence humaine.

    Mesurer au minimum :

        Exact Match sur quelques cas simples ;

        similarité sémantique ;

        ou notation manuelle correcte / partielle / incorrecte.

    Si tu as le temps, ajouter un script Ragas.

Livrables :

    fichier evaluation_dataset.jsonl ;

    script evaluate_rag.py ;

    tableau de résultats.
    OpenClassrooms insiste sur la constitution d’un dataset d’évaluation, et Ragas peut servir à automatiser des métriques de pertinence et de fidélité si tu veux aller un peu plus loin sans surcomplexifier.

7. Tests

    Tester la récupération de données.

    Tester la construction de l’index.

    Tester la recherche vectorielle.

    Tester ask().

    Tester les routes API et la gestion des erreurs, notamment question vide ou payload invalide.

Livrables :

    dossier tests/ ;

    exécution simple avec pytest.
    À ce stade, des tests basiques mais ciblés suffisent largement pour un projet scolaire.

Livraison
8. Dockerisation

    Créer un Dockerfile simple.

    Permettre de lancer l’API localement dans un conteneur.

    Vérifier que le projet repart depuis une installation propre.

    Éviter d’embarquer la clé API dans l’image.

Livrables :

    Dockerfile ;

    commande de build et de run dans le README ;

    démonstration locale fonctionnelle.
    L’objectif ici est surtout de prouver que la solution peut être relancée facilement, pas de faire une infra de production.

9. Documentation et soutenance

    Finaliser le README.md : objectif, structure, installation, configuration .env, lancement API, rebuild index, évaluation.

    Préparer 10 à 15 slides.

    Préparer 2 ou 3 scénarios de démo réalistes.

    Préparer une explication simple de ce qu’est un RAG pour un public non technique.

Slides suggérées :

    Contexte et besoin.

    Objectif du POC.

    Données OpenAgenda.

    Environnement reproductible.

    Architecture RAG.

    Construction de l’index FAISS.

    Génération des réponses avec Mistral.

    API REST.

    Évaluation.

    Résultats.

    Démo Docker.

    Limites et perspectives.

Ordre conseillé

Voici l’ordre le plus simple à suivre :

    Environnement + README.

    Collecte et nettoyage des données.

    Chunking + embeddings + FAISS.

    Fonction RAG ask().

    API FastAPI.

    Jeu de test annoté + évaluation.

    Tests unitaires.

    Docker.

    Slides + répétition de la démo.

Niveau attendu

Pour rester dans un cadre scolaire, vise un projet :

    fonctionnel plutôt qu’ultra-optimisé ;

    lisible plutôt qu’architecturé comme un produit SaaS ;

    justifié dans ses choix techniques ;

    démo-able sans dépendances fragiles.

Une bonne cible est : quelques centaines d’événements, une API qui répond proprement, un rebuild manuel de l’index, un petit protocole d’évaluation, et une documentation claire.

Je peux maintenant te transformer ça en checklist de rendu sur 5 jours ou en plan de dépôt GitHub prêt à copier.