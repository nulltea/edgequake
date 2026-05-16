'use client';

import {
    WorkspaceLoading,
    WorkspaceNotFound,
    WorkspaceRedirecting,
} from '@/components/workspace/workspace-deeplink-states';
import { useWorkspaceSlugResolver } from '@/hooks/use-workspace-slug-resolver';
import { useParams, useRouter } from 'next/navigation';
import { useEffect } from 'react';

/**
 * Workspace archive deeplink — sets workspace context and redirects to /archive.
 */
export default function WorkspaceArchivePage() {
  const params = useParams();
  const router = useRouter();
  const slug = params?.slug as string;

  const { workspace, isLoading, error, isReady } = useWorkspaceSlugResolver(slug);

  useEffect(() => {
    if (isReady) {
      router.push('/archive');
    }
  }, [isReady, router]);

  if (isLoading) {
    return <WorkspaceLoading context="workspace archive" />;
  }

  if (error || !workspace) {
    return (
      <WorkspaceNotFound
        slug={slug}
        fallbackHref="/archive"
        fallbackLabel="Go to Archive"
      />
    );
  }

  return <WorkspaceRedirecting />;
}
