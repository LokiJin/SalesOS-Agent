import hashlib
import sqlite3
from datetime import datetime
from typing import List, Dict, Set, Optional
from pathlib import Path

class RAGMetadataManager:
    """
    Manages metadata for incremental RAG updates using SQLite.
    Tracks file hashes and their associated chunk IDs in ChromaDB.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize metadata manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_metadata (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    ingested_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_metadata (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    FOREIGN KEY (file_path) REFERENCES file_metadata(file_path)
                        ON DELETE CASCADE
                )
            """)
            
            # Index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_file_path 
                ON chunk_metadata(file_path)
            """)
            
            conn.commit()
    
    def compute_file_hash(self, file_path: str) -> str:
        """
        Compute SHA256 hash of file contents.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hex digest of file hash
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error hashing {file_path}: {e}")
            return ""
    
    def get_stored_hash(self, file_path: str) -> Optional[str]:
        """
        Get stored hash for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Stored hash or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_hash FROM file_metadata WHERE file_path = ?",
                (file_path,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_chunk_ids(self, file_path: str) -> List[str]:
        """
        Get all chunk IDs associated with a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of chunk IDs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT chunk_id FROM chunk_metadata WHERE file_path = ? ORDER BY chunk_index",
                (file_path,)
            )
            return [row[0] for row in cursor.fetchall()]
    
    def store_file_metadata(self, file_path: str, file_hash: str, chunk_ids: List[str], file_type: str = "txt"):
        """
        Store metadata for a newly ingested file.
        
        Args:
            file_path: Path to file
            file_hash: Hash of file contents
            chunk_ids: List of chunk IDs in ChromaDB
            file_type: File extension/type ('pdf', 'docx', etc)
        """
        now = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Store file metadata
            conn.execute("""
                INSERT OR REPLACE INTO file_metadata 
                (file_path, file_hash, file_type, chunk_count, ingested_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_path, file_hash, file_type, len(chunk_ids), now, now))
            
            # Delete old chunk metadata (in case of update)
            conn.execute(
                "DELETE FROM chunk_metadata WHERE file_path = ?",
                (file_path,)
            )
            
            # Store chunk metadata
            chunk_data = [
                (chunk_id, file_path, idx)
                for idx, chunk_id in enumerate(chunk_ids)
            ]
            conn.executemany("""
                INSERT INTO chunk_metadata (chunk_id, file_path, chunk_index)
                VALUES (?, ?, ?)
            """, chunk_data)
            
            conn.commit()
    
    def delete_file_metadata(self, file_path: str) -> List[str]:
        """
        Delete metadata for a file and return its chunk IDs.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of chunk IDs that were deleted
        """
        chunk_ids = self.get_chunk_ids(file_path)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM file_metadata WHERE file_path = ?",
                (file_path,)
            )
            # CASCADE will handle chunk_metadata deletion
            conn.commit()
        
        return chunk_ids
    
    def get_all_tracked_files(self) -> Set[str]:
        """
        Get set of all files currently tracked in metadata.
        
        Returns:
            Set of file paths
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT file_path FROM file_metadata")
            return {row[0] for row in cursor.fetchall()}
    
    def get_stats(self) -> Dict:
        """
        Get statistics about tracked files.
        
        Returns:
            Dictionary with stats
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as file_count,
                    SUM(chunk_count) as total_chunks
                FROM file_metadata
            """)
            row = cursor.fetchone()
            
            return {
                "file_count": row[0] or 0,
                "total_chunks": row[1] or 0
            }