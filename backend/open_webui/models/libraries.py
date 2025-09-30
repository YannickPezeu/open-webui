# src/apps/webui/models/libraries.py
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Table
from datetime import datetime
from open_webui.internal.db import Base

# Table de liaison
library_group = Table(
    'library_group',
    Base.metadata,
    Column('library_id', String, ForeignKey('library.id'), primary_key=True),
    Column('group_id', String, ForeignKey('group.id'), primary_key=True)
)

class Library(Base):
    __tablename__ = "library"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    user_id = Column(String)  # Pas de ForeignKey si User n'a pas de relationship
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # ❌ PAS de relationship() vers Group
    # Parce que Group utilise un pattern différent


# Classe utilitaire pour les opérations (comme GroupTable)
class LibraryTable:
    def get_library_groups(self, library_id: str, db) -> list[str]:
        """Récupère les group_ids d'une library"""
        result = db.execute(
            library_group.select().where(
                library_group.c.library_id == library_id
            )
        ).fetchall()
        return [row.group_id for row in result]
    
    def add_groups_to_library(self, library_id: str, group_ids: list[str], db):
        """Ajoute des groupes à une library"""
        for group_id in group_ids:
            db.execute(
                library_group.insert().values(
                    library_id=library_id,
                    group_id=group_id
                )
            )

Libraries = LibraryTable()