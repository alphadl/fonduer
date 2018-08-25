import logging

from sqlalchemy.sql import select

from fonduer.candidates.models import Candidate
from fonduer.features.features import get_all_feats
from fonduer.features.models import Feature, FeatureKey
from fonduer.meta import new_sessionmaker
from fonduer.utils.udf import UDF, UDFRunner

logger = logging.getLogger(__name__)


class FAnnotator(UDFRunner):
    """An operator to add Feature Annotations to Candidates."""

    def __init__(self, candidate_classes):
        """Initialize the FAnnotator."""
        super(FAnnotator, self).__init__(
            FAnnotatorUDF, candidate_classes=candidate_classes
        )
        self.candidate_classes = candidate_classes

    def apply(
        self,
        split=0,
        replace_key_set=True,
        update_keys=False,
        update_values=True,
        **kwargs
    ):
        """Call the FAnnotatorUDF."""
        # Get the cids based on the split, and also the count
        Session = new_sessionmaker()
        session = Session()

        for candidate_class in self.candidate_classes:
            # NOTE: In the current UDFRunner implementation, we load all these into
            # memory and fill a multiprocessing JoinableQueue with them before
            # starting... so might as well load them here and pass in. Also, if we
            # try to pass in a query iterator instead, with AUTOCOMMIT on, we get a
            # TXN error...
            candidates = (
                session.query(candidate_class)
                .filter(candidate_class.split == split)
                .all()
            )
            cids_count = len(candidates)
            if cids_count == 0:
                logger.warning(
                    "No {} candidates in split {}".format(
                        candidate_class.__name__, split
                    )
                )
                continue

            # Parallelize on Candidates
            super(FAnnotator, self).apply(candidates, split=split, **kwargs)

        return self.load_matrices(session, split=split)

    def clear(
        self,
        session,
        split=0,
        replace_key_set=True,
        update_keys=False,
        update_values=True,
        **kwargs
    ):
        """Delete Features of each class from the database."""
        for candidate_class in self.candidate_classes:
            logger.info(
                "Clearing {} Features (split {})".format(
                    candidate_class.__tablename__, split
                )
            )
            query = session.query(candidate_class)
            if not replace_key_set:
                sub_query = (
                    session.query(Candidate.id)
                    .filter(Candidate.split == split)
                    .subquery()
                )
                query = query.filter(candidate_class.id.in_(sub_query))
            query.delete(synchronize_session="fetch")

            # If we are creating a new key set, delete all old annotation keys
            if replace_key_set:
                query = session.query(FeatureKey)
                query.delete(synchronize_session="fetch")

    def clear_all(self, session, **kwargs):
        """Delete all Features."""
        logger.info("Clearing ALL Features.")
        session.query(Feature).delete()

    def load_matrices(self, session, split=0, **kwargs):
        """Load sparse matrix of Features for each candidate_class."""

        for x in self.candidate_classes:
            yield x.__tablename__


class FAnnotatorUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(self, candidate_classes, **kwargs):
        """Initialize the FAnnotatorUDF."""
        self.candidate_classes = (
            candidate_classes
            if isinstance(candidate_classes, (list, tuple))
            else [candidate_classes]
        )
        super(FAnnotatorUDF, self).__init__(**kwargs)

    def _get_FeatureKey(self, key):
        """Construct a FeatureKey from the key."""
        key_args = {"name": key}
        return FeatureKey(**key_args)

    def apply(self, candidate, clear, **kwargs):
        """Extract candidates from the given Context.

        :param context: A document to process.
        :param clear: Whether or not to clear the existing database entries.
        :param split: Which split to use.
        """
        logger.debug("Candidate: {}".format(candidate))
        feature_args = {"candidate_id": candidate.id}
        keys = []
        values = []
        for cid, key_name, feature in get_all_feats(candidate):
            if feature == 0:
                continue
            # Construct FeatureKey and yield
            # Check for existence
            if (
                not self.session.query(FeatureKey)
                .filter(FeatureKey.name == key_name)
                .first()
            ):
                yield self._get_FeatureKey(key_name)
            keys.append(key_name)
            values.append(feature)

        # Assemble feature arguments
        feature_args["keys"] = keys
        feature_args["values"] = values

        # Checking for existence
        if not clear:
            q = select([Feature.id])
            for key, value in list(feature_args.items()):
                q = q.where(getattr(Feature, key) == value)
            feature_id = self.session.execute(q).first()
            if feature_id is not None:
                return

        # Add Candidate to session
        yield Feature(**feature_args)
