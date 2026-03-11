package raft

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"sync"
)

const (
	recordTypeEntry byte = 0x01
	recordTypeState byte = 0x02

	walFileName   = "raft.wal"
	stateFileName = "raft.state"
	syncInterval  = 32
)

// PersistedState holds the durable Raft state.
type PersistedState struct {
	Term     uint64 `json:"term"`
	VotedFor string `json:"voted_for"`
}

// WAL provides durable storage for Raft log entries and state.
type WAL struct {
	mu       sync.Mutex
	dir      string
	file     *os.File
	writer   *bufio.Writer
	writeN   int
	crcTable *crc32.Table
}

// NewWAL creates or opens a WAL in the given directory.
func NewWAL(dir string) (*WAL, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("create WAL dir: %w", err)
	}

	path := filepath.Join(dir, walFileName)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return nil, fmt.Errorf("open WAL file: %w", err)
	}

	return &WAL{
		dir:      dir,
		file:     f,
		writer:   bufio.NewWriterSize(f, 64*1024),
		crcTable: crc32.MakeTable(crc32.Castagnoli),
	}, nil
}

// Append writes a log entry to the WAL.
func (w *WAL) Append(entry LogEntry) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	payload, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("marshal entry: %w", err)
	}

	if err := w.writeRecord(recordTypeEntry, payload); err != nil {
		return err
	}

	w.writeN++
	if w.writeN%syncInterval == 0 {
		return w.sync()
	}
	return nil
}

// PersistState durably records the current term and votedFor.
func (w *WAL) PersistState(term uint64, votedFor string) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	state := PersistedState{Term: term, VotedFor: votedFor}
	path := filepath.Join(w.dir, stateFileName)
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal state: %w", err)
	}

	// Atomic write: temp file then rename
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("write state tmp: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("rename state: %w", err)
	}
	return nil
}

// LoadState reads the persisted Raft state from disk.
func (w *WAL) LoadState() (*PersistedState, error) {
	path := filepath.Join(w.dir, stateFileName)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read state: %w", err)
	}

	var state PersistedState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("unmarshal state: %w", err)
	}
	return &state, nil
}

// LoadEntries reads all log entries from the WAL.
func (w *WAL) LoadEntries() ([]LogEntry, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.writer.Flush(); err != nil {
		return nil, fmt.Errorf("flush before read: %w", err)
	}
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek: %w", err)
	}

	reader := bufio.NewReader(w.file)
	var entries []LogEntry

	for {
		recType, payload, err := w.readRecord(reader)
		if err == io.EOF {
			break
		}
		if err != nil {
			break // Truncated record at end
		}
		if recType == recordTypeEntry {
			var entry LogEntry
			if err := json.Unmarshal(payload, &entry); err != nil {
				return nil, fmt.Errorf("unmarshal entry: %w", err)
			}
			entries = append(entries, entry)
		}
	}

	if _, err := w.file.Seek(0, io.SeekEnd); err != nil {
		return nil, fmt.Errorf("seek to end: %w", err)
	}
	return entries, nil
}

// Truncate removes all entries after the given index.
func (w *WAL) Truncate(afterIndex uint64) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	entries, err := w.loadEntriesLocked()
	if err != nil {
		return err
	}

	var kept []LogEntry
	for _, e := range entries {
		if e.Index <= afterIndex {
			kept = append(kept, e)
		}
	}
	return w.rewriteWAL(kept)
}

// Close flushes and closes the WAL.
func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if err := w.writer.Flush(); err != nil {
		return err
	}
	if err := w.file.Sync(); err != nil {
		return err
	}
	return w.file.Close()
}

// -- Internal --

func (w *WAL) writeRecord(recType byte, payload []byte) error {
	crc := crc32.New(w.crcTable)
	crc.Write([]byte{recType})
	crc.Write(payload)
	checksum := crc.Sum32()

	header := make([]byte, 9)
	binary.LittleEndian.PutUint32(header[0:4], checksum)
	binary.LittleEndian.PutUint32(header[4:8], uint32(len(payload)))
	header[8] = recType

	if _, err := w.writer.Write(header); err != nil {
		return fmt.Errorf("write header: %w", err)
	}
	if _, err := w.writer.Write(payload); err != nil {
		return fmt.Errorf("write payload: %w", err)
	}
	return nil
}

func (w *WAL) readRecord(reader *bufio.Reader) (byte, []byte, error) {
	header := make([]byte, 9)
	if _, err := io.ReadFull(reader, header); err != nil {
		return 0, nil, err
	}

	checksum := binary.LittleEndian.Uint32(header[0:4])
	length := binary.LittleEndian.Uint32(header[4:8])
	recType := header[8]

	if length > 64*1024*1024 {
		return 0, nil, fmt.Errorf("record too large: %d bytes", length)
	}

	payload := make([]byte, length)
	if _, err := io.ReadFull(reader, payload); err != nil {
		return 0, nil, err
	}

	crc := crc32.New(w.crcTable)
	crc.Write([]byte{recType})
	crc.Write(payload)
	if crc.Sum32() != checksum {
		return 0, nil, fmt.Errorf("CRC mismatch: record corrupted")
	}

	return recType, payload, nil
}

func (w *WAL) sync() error {
	if err := w.writer.Flush(); err != nil {
		return fmt.Errorf("flush: %w", err)
	}
	if err := w.file.Sync(); err != nil {
		return fmt.Errorf("fsync: %w", err)
	}
	return nil
}

func (w *WAL) loadEntriesLocked() ([]LogEntry, error) {
	if err := w.writer.Flush(); err != nil {
		return nil, err
	}
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	reader := bufio.NewReader(w.file)
	var entries []LogEntry
	for {
		recType, payload, err := w.readRecord(reader)
		if err != nil {
			break
		}
		if recType == recordTypeEntry {
			var entry LogEntry
			if err := json.Unmarshal(payload, &entry); err != nil {
				continue
			}
			entries = append(entries, entry)
		}
	}
	return entries, nil
}

func (w *WAL) rewriteWAL(entries []LogEntry) error {
	w.writer.Flush()
	w.file.Close()

	path := filepath.Join(w.dir, walFileName)
	newPath := path + ".new"

	f, err := os.Create(newPath)
	if err != nil {
		return fmt.Errorf("create new WAL: %w", err)
	}

	w.file = f
	w.writer = bufio.NewWriterSize(f, 64*1024)

	for _, entry := range entries {
		payload, err := json.Marshal(entry)
		if err != nil {
			return err
		}
		if err := w.writeRecord(recordTypeEntry, payload); err != nil {
			return err
		}
	}

	if err := w.sync(); err != nil {
		return err
	}

	if err := os.Rename(newPath, path); err != nil {
		return fmt.Errorf("rename WAL: %w", err)
	}

	w.file.Close()
	w.file, err = os.OpenFile(path, os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("reopen WAL: %w", err)
	}
	w.writer = bufio.NewWriterSize(w.file, 64*1024)
	return nil
}
