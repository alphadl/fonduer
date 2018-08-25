from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects import postgresql

from fonduer.meta import Meta
from fonduer.utils.models.annotation import AnnotationKeyMixin, AnnotationMixin

_meta = Meta.init()


class GoldLabelKey(AnnotationKeyMixin, _meta.Base):
    pass


class GoldLabel(AnnotationMixin, _meta.Base):
    """A separate class for labels from human annotators or other gold standards."""

    value = Column(Integer, nullable=False)


class LabelKey(AnnotationKeyMixin, _meta.Base):
    pass


class Label(AnnotationMixin, _meta.Base):
    """A discrete label associated with a Candidate, indicating a prediction value.

    Labels are used to represent the output of labeling functions. A Label's
    annotation key identifies the labeling function that provided the Label.
    """

    values = Column(postgresql.ARRAY(Integer), nullable=False)


class StableLabel(_meta.Base):
    """A stable table for preserving human annotations.

    A special secondary table for preserving labels created by *human
    annotators* in a stable format that does not cascade, and is independent of
    the Candidate ids.
    """

    __tablename__ = "stable_label"
    context_stable_ids = Column(
        String, primary_key=True
    )  # ~~ delimited list of the context stable ids
    annotator_name = Column(String, primary_key=True)
    split = Column(Integer, default=0)
    value = Column(Integer, nullable=False)

    def __repr__(self):
        return "%s (%s : %s)" % (
            self.__class__.__name__,
            self.annotator_name,
            self.value,
        )
