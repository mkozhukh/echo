package echo

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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
		return fmt.Errorf("status code: %d", resp.StatusCode)
	}

	return json.NewDecoder(resp.Body).Decode(responsePtr)
}
