package echo

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type RequestInit func(*http.Request)

// callHTTPAPI is a generic function that makes HTTP requests and decodes responses
func callHTTPAPI(ctx context.Context, url string, init RequestInit, body any, responsePtr any) error {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	if err != nil {
		return err
	}

	init(req)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("status code: %d, body: %s", resp.StatusCode, string(body))
	}

	err = json.NewDecoder(resp.Body).Decode(responsePtr)
	if err != nil {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to decode response: %w, body: %s", err, string(body))
	}

	return nil
}

// streamHTTPAPI makes streaming HTTP requests and returns the response body
func streamHTTPAPI(ctx context.Context, url string, init RequestInit, body any) (io.ReadCloser, error) {
	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	init(req)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status code: %d, body: %s", resp.StatusCode, string(body))
	}

	return resp.Body, nil
}

// SSEMessage represents a parsed SSE message
type SSEMessage struct {
	Event string
	Data  []byte
}

var eventPrefix = []byte("event: ")
var dataPrefix = []byte("data: ")
var doneMarker = []byte("[DONE]")
var emptyLine = []byte("")

// parseSSEStream parses Server-Sent Events stream and calls handler for each complete message
func parseSSEStream(respBody io.ReadCloser, handler func(SSEMessage) error) error {
	defer respBody.Close()

	var buffer bytes.Buffer
	reader := bufio.NewReader(respBody)
	var currentEvent string

	for {
		line, err := reader.ReadBytes('\n')
		if err == io.EOF {
			// Process any remaining data in buffer
			if buffer.Len() > 0 {
				msg := SSEMessage{Event: currentEvent, Data: buffer.Bytes()}
				if err := handler(msg); err != nil {
					return err
				}
			}
			break
		}
		if err != nil {
			return fmt.Errorf("read error: %w", err)
		}

		// Check for double newline (message separator)
		if bytes.Equal(bytes.TrimSpace(line), emptyLine) {
			// End of message, process buffer contents if we have data
			if buffer.Len() > 0 {
				msg := SSEMessage{Event: currentEvent, Data: buffer.Bytes()}
				if err := handler(msg); err != nil {
					return err
				}
				buffer.Reset()
				currentEvent = ""
			}
			continue
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		// Parse SSE fields
		if bytes.HasPrefix(line, eventPrefix) {
			currentEvent = string(bytes.TrimPrefix(line, eventPrefix))
		} else if bytes.HasPrefix(line, dataPrefix) {
			data := bytes.TrimPrefix(line, dataPrefix)
			buffer.Write(data)
		}
	}

	return nil
}
