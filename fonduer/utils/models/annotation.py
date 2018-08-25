from sqlalchemy import Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import backref, relationship

from fonduer.meta import Meta
from fonduer.utils.utils import camel_to_under

# Grab pointer to global metadata
_meta = Meta.init()


class AnnotationKeyMixin(object):
    """Mixin class for defining annotation key tables.

    An AnnotationKey is the unique name associated with a set of Annotations,
    corresponding e.g. to a single labeling or feature function.  An
    AnnotationKey may have an associated weight (Parameter) associated with it.
    """

    @declared_attr
    def __tablename__(cls):
        return camel_to_under(cls.__name__)

    @declared_attr
    def id(cls):
        return Column(Integer, primary_key=True)

    @declared_attr
    def name(cls):
        return Column(String, nullable=False)

    @declared_attr
    def group(cls):
        return Column(Integer, nullable=False, default=0)

    @declared_attr
    def __table_args__(cls):
        return (UniqueConstraint("name", "group"),)

    def __repr__(self):
        return str(self.__class__.__name__) + " (" + str(self.name) + ")"


class AnnotationMixin(object):
    """Mixin class for defining annotation tables.

    An annotation is a value associated with a Candidate. Examples include
    labels, features, and predictions. New types of annotations can be defined
    by creating an annotation class and corresponding annotation, for example:

    .. code-block:: python

        from fonduer.supervision.models import AnnotationMixin
        from fonduer.meta import Meta

        class NewAnnotation(AnnotationMixin, Meta.Base):
            value = Column(Float, nullable=False)


        # The entire storage schema, including NewAnnotation, can now be
        # initialized with the following import
        import fonduer.models

    The annotation class should include a Column attribute named value.
    """

    @declared_attr
    def __tablename__(cls):
        return camel_to_under(cls.__name__)

    # The key is the "name" or "type" of the Annotation- e.g. the name of a
    # feature, lf, or of a human annotator
    @declared_attr
    def keys(cls):
        return Column(postgresql.ARRAY(String), nullable=False)

    # Every annotation is with respect to a candidate
    @declared_attr
    def candidate_id(cls):
        return Column(
            "candidate_id",
            Integer,
            ForeignKey("candidate.id", ondelete="CASCADE"),
            primary_key=True,
        )

    @declared_attr
    def candidate(cls):
        return relationship(
            "Candidate",
            backref=backref(
                camel_to_under(cls.__name__) + "s",
                cascade="all, delete-orphan",
                cascade_backrefs=False,
            ),
            cascade_backrefs=False,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.type)
            + "): ("
            + str(self.keys)
            + " = "
            + str(self.values)
            + ")"
        )
