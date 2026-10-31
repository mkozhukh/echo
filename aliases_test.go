package echo_test

import (
	"strings"
	"testing"

	"github.com/mkozhukh/echo"
	"github.com/mkozhukh/echo/pricing"
)

// Every alias must point to a model known to the built-in price table, which
// tracks models.dev; a miss usually means the alias targets a retired model.
func TestAliasesArePriced(t *testing.T) {
	prices := pricing.Default()
	for alias, model := range echo.Aliases {
		if !strings.HasPrefix(model, strings.SplitN(alias, "/", 2)[0]+"/") {
			t.Errorf("alias %s -> %s: provider mismatch", alias, model)
		}
		if _, ok := prices.Get(model); !ok {
			t.Errorf("alias %s -> %s: model not in price table", alias, model)
		}
	}
}
