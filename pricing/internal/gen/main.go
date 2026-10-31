// Command gen refreshes the embedded price snapshot (prices.json) from models.dev.
//
// It keeps only the providers echo talks to and only the cost fields used for
// text pricing. Run it via `go generate ./pricing`.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"sort"
)

var providers = []string{"openai", "anthropic", "google", "xai", "openrouter"}

var costFields = []string{"input", "output", "cache_read", "cache_write", "reasoning", "tiers"}

type provider struct {
	Models map[string]struct {
		Cost map[string]json.RawMessage `json:"cost"`
	} `json:"models"`
}

func main() {
	src := flag.String("src", "https://models.dev/api.json", "models.dev api.json URL")
	out := flag.String("o", "prices.json", "output file")
	flag.Parse()

	if err := run(*src, *out); err != nil {
		fmt.Fprintln(os.Stderr, "gen:", err)
		os.Exit(1)
	}
}

func run(src, out string) error {
	resp, err := http.Get(src)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", src, resp.Status)
	}

	var all map[string]provider
	if err := json.NewDecoder(resp.Body).Decode(&all); err != nil {
		return err
	}

	type model struct {
		Cost map[string]json.RawMessage `json:"cost"`
	}
	trimmed := map[string]map[string]map[string]model{}
	count := 0
	for _, name := range providers {
		p, ok := all[name]
		if !ok {
			return fmt.Errorf("provider %q not found in %s", name, src)
		}
		models := map[string]model{}
		for id, m := range p.Models {
			cost := map[string]json.RawMessage{}
			for _, f := range costFields {
				if v, ok := m.Cost[f]; ok {
					cost[f] = v
				}
			}
			if _, ok := cost["input"]; !ok {
				continue
			}
			models[id] = model{Cost: cost}
		}
		trimmed[name] = map[string]map[string]model{"models": models}
		count += len(models)
	}

	// stable output: encoding/json sorts map keys
	data, err := json.MarshalIndent(trimmed, "", " ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(out, append(data, '\n'), 0o644); err != nil {
		return err
	}

	names := append([]string(nil), providers...)
	sort.Strings(names)
	fmt.Printf("gen: wrote %d models from %v to %s\n", count, names, out)
	return nil
}
