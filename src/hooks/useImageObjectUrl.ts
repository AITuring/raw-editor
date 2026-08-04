import { useCallback, useEffect, useRef, useState } from 'react';
import { revokeImageObjectUrl, revokeImageObjectUrlLater } from '../utils/imageObjectUrl';

export function useImageObjectUrl() {
  const [url, setUrl] = useState<string | null>(null);
  const urlRef = useRef<string | null>(null);

  const replace = useCallback((nextUrl: string | null) => {
    const previousUrl = urlRef.current;
    if (previousUrl === nextUrl) return;

    urlRef.current = nextUrl;
    setUrl(nextUrl);
    revokeImageObjectUrlLater(previousUrl);
  }, []);

  const clear = useCallback(() => replace(null), [replace]);

  useEffect(
    () => () => {
      revokeImageObjectUrl(urlRef.current);
      urlRef.current = null;
    },
    [],
  );

  return { url, replace, clear };
}
